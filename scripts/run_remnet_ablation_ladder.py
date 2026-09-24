#!/usr/bin/env python3
"""Orchestrate the registered REM-Net ablation ladder (experiments/specs/q1_remnet_ablation_ladder.json).

Four rungs, each changing one variable against the rung below, 11 strict-LOSO folds,
seed 42. Every fold trains on train, selects one checkpoint on validation, and only
then opens the held-out test subject once. Completed folds are skipped, so an
interrupted run resumes where it stopped.

  R0  raw RCG input, waveform head only
  R1  + WSST representation
  R2  + anchor supervision, head only (backbone detached)
  R3  + joint optimisation  (the frozen L3 configuration, rerun as a parity arm)

Usage:
  python scripts/run_remnet_ablation_ladder.py --preflight_only
  python scripts/run_remnet_ablation_ladder.py
  python scripts/run_remnet_ablation_ladder.py --aggregate_only
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev

import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.atomic_state import atomic_csv, atomic_json, atomic_text  # noqa: E402

TAG = "q1_remnet_ablation_ladder_seed42"
SPEC = ROOT / "experiments" / "specs" / "q1_remnet_ablation_ladder.json"
REGISTRY = ROOT / "experiments_mmecg" / "EXPERIMENT_REGISTRY.csv"
OUT = ROOT / "experiments_mmecg" / TAG
RUNGS = ("R0", "R1", "R2", "R3")
FOLDS = tuple(range(1, 12))
METRICS = ("pcc", "rmse", "nrmse", "mae", "r2", "amplitude_variance_ratio",
           "f1_100ms", "f1_150ms", "event_f1_100ms", "event_f1_150ms",
           "rr_error_ms", "rr_coverage", "qrs_region_mae", "stt_region_mae")


def capture(*cmd: str) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()


def require_preflight() -> tuple[str, str]:
    spec = json.loads(SPEC.read_text())
    if not str(spec.get("status", "")).startswith("APPROVED"):
        raise SystemExit(f"Spec not approved: {spec.get('status')}")
    dirty = "\n".join(line for line in capture(
        "git", "status", "--short", "--untracked-files=no").splitlines()
        if not line.endswith("experiments_mmecg/EXPERIMENT_REGISTRY.csv"))
    if dirty:
        raise SystemExit("Refusing formal launch with uncommitted tracked files:\n" + dirty)
    split_root = Path("/home/qhh2237/Datasets/MMECG/processed/loso_v2")
    pool = sorted((split_root.parent / "wsst8s_subject_pool").glob("subject_*.h5"))
    if len(pool) != 11:
        raise SystemExit(f"Subject pool incomplete: {len(pool)} files")
    for fold in FOLDS:
        for split in ("train", "val", "test"):
            path = split_root / f"fold_{fold:02d}" / f"{split}.h5"
            if not path.exists():
                raise SystemExit(f"Missing required input: {path}")
            if split == "train":
                with h5py.File(path, "r") as h:
                    if "rcg" not in h:
                        raise SystemExit(f"R0 needs the rcg dataset: {path}")
    return capture("git", "rev-parse", "HEAD"), capture("git", "branch", "--show-current")


def update_registry(commit: str, branch: str, status: str) -> None:
    lines = REGISTRY.read_text().splitlines()
    fields = next(csv.reader([lines[0]]))
    values = {
        "tag": TAG, "dataset": "MMECG public",
        "protocol": "loso_v2 complete 11-fold strict LOSO; validation selection then one-time test",
        "category": "ablation", "publishable": "false", "commit_id": commit, "branch": branch,
        "role": "REM-Net ablation ladder R0 (RCG) -> R1 (+WSST) -> R2 (+anchor head, detached) "
                "-> R3 (+joint, = frozen L3)",
        "status": status,
        "reason": "Approved by the user 2026-09-25; single seed 42, one-table ablation only "
                  "(METRIC_CONTRACT: >= 3 seeds for headline claims).",
    }
    buf = io.StringIO()
    csv.DictWriter(buf, fieldnames=fields, lineterminator="").writerow(values)
    row = buf.getvalue()
    for i, line in enumerate(lines[1:], start=1):
        if next(csv.reader([line]))[0] == TAG:
            lines[i] = row
            break
    else:
        lines.append(row)
    atomic_text(REGISTRY, "\n".join(lines) + "\n")


def run_rung(rung: str, device: str) -> None:
    for fold in FOLDS:
        fold_dir = OUT / rung / f"fold_{fold:02d}"
        state_path = fold_dir / "run_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        if state and state.get("stage") == "COMPLETE":
            if not (fold_dir / "test_metrics.json").exists():
                raise SystemExit(f"Corrupt COMPLETE fold without metrics: {fold_dir}")
            print(f"[skip] {rung} fold {fold:02d} complete", flush=True)
            continue
        cmd = [sys.executable, "scripts/train_continuous_ecg_pilot.py",
               "--arm", "B1", "--loss_arm", "L3", "--ladder_rung", rung, "--fold", str(fold),
               "--split_root", "loso_v2", "--exp_tag", TAG, "--final_test",
               "--wsst_source", "subject_pool", "--device", device, "--seed_base", "42"]
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


def aggregate() -> None:
    summary = {"experiment_id": TAG, "aggregation": "fold-macro over 11 held-out test subjects",
               "rungs": {}}
    for rung in RUNGS:
        rows = []
        for fold in FOLDS:
            path = OUT / rung / f"fold_{fold:02d}" / "test_metrics.json"
            if not path.exists():
                print(f"[aggregate] {rung} incomplete, missing {path.name} in fold {fold:02d}")
                rows = []
                break
            m = json.loads(path.read_text())["metrics"]
            rows.append({"fold": fold, **{k: m.get(k) for k in METRICS}})
        if not rows:
            continue
        atomic_csv(OUT / rung / "test_fold_metrics.csv", list(rows[0]), rows)
        summary["rungs"][rung] = {
            k: {"mean": mean(float(r[k]) for r in rows if r[k] is not None),
                "std_population": pstdev(float(r[k]) for r in rows if r[k] is not None)}
            for k in METRICS}
    atomic_json(OUT / "ladder_summary.json", summary)
    print(f"wrote {OUT / 'ladder_summary.json'} ({len(summary['rungs'])} complete rungs)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--preflight_only", action="store_true")
    ap.add_argument("--aggregate_only", action="store_true")
    ap.add_argument("--rungs", nargs="+", choices=RUNGS, default=list(RUNGS))
    args = ap.parse_args()
    if args.aggregate_only:
        aggregate()
        return
    commit, branch = require_preflight()
    update_registry(commit, branch, "preflight-passed" if args.preflight_only else "running")
    if args.preflight_only:
        print(f"PASS: {TAG} at {commit} on {branch}")
        return
    try:
        for rung in args.rungs:
            run_rung(rung, args.device)
        aggregate()
    except Exception:
        update_registry(commit, branch, "failed; inspect per-fold artifacts")
        raise
    update_registry(commit, branch, "completed; awaiting ladder audit")


if __name__ == "__main__":
    main()
