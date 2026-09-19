#!/usr/bin/env python3
"""Orchestrate the registered 11-fold continuous REM-Net final candidate.

This entry point does not alter the model or select from test results. Each
fold trains on train, selects one checkpoint on validation, and only then opens
the held-out test subject once. Existing completed folds are resumable.
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
SEED_BASE = 42
TAG = "q1_remnet_continuous_strict_loso_seed42"
SPEC = ROOT / "experiments" / "specs" / f"{TAG}.json"
REGISTRY = ROOT / "experiments_mmecg" / "EXPERIMENT_REGISTRY.csv"
OUT = ROOT / "experiments_mmecg" / TAG


def configure(seed_base: int) -> None:
    """Point the runner at one registered seed of the continuous REM-Net benchmark."""
    global SEED_BASE, TAG, SPEC, OUT
    SEED_BASE = seed_base
    TAG = f"q1_remnet_continuous_strict_loso_seed{seed_base}"
    SPEC = ROOT / "experiments" / "specs" / f"{TAG}.json"
    OUT = ROOT / "experiments_mmecg" / TAG
FOLDS = tuple(range(1, 12))
METRICS = (
    "pcc", "rmse", "nrmse", "mae", "r2", "amplitude_variance_ratio",
    "above_10hz_energy_ratio", "f1_50ms", "f1_100ms", "f1_150ms",
    "event_f1_50ms", "event_f1_100ms", "event_f1_150ms", "rr_error_ms",
    "hr_error_bpm", "rr_coverage", "qrs_region_mae", "stt_region_mae",
)


def capture(*cmd: str) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()


def require_preflight() -> tuple[str, str]:
    spec = json.loads(SPEC.read_text())
    if spec.get("status") not in {"DATA_READY", "PREFLIGHT_PASSED"}:
        raise SystemExit(f"Unexpected spec state: {spec.get('status')}")
    dirty = "\n".join(line for line in capture(
        "git", "status", "--short", "--untracked-files=no").splitlines()
        # the registry is this runner's own bookkeeping file
        if not line.endswith("experiments_mmecg/EXPERIMENT_REGISTRY.csv"))
    if dirty:
        raise SystemExit("Refusing formal launch with uncommitted tracked files:\n" + dirty)
    split_root = Path("/home/qhh2237/Datasets/MMECG/processed/loso_v2")
    pool_root = split_root.parent / "wsst8s_subject_pool"
    pool_files = sorted(pool_root.glob("subject_*.h5"))
    if len(pool_files) != 11:
        raise SystemExit(f"Subject pool incomplete: expected 11 files, found {len(pool_files)}")
    subjects = set()
    for path in pool_files:
        with h5py.File(path, "r") as h:
            if h.attrs.get("support") != "full_window_samples_0_1600":
                raise SystemExit(f"Invalid pool support: {path}")
            subjects.add(int(h.attrs["subject_id"]))
    if len(subjects) != 11:
        raise SystemExit(f"Subject pool IDs are not unique: {sorted(subjects)}")
    for fold in FOLDS:
        for split in ("train", "val", "test"):
            path = split_root / f"fold_{fold:02d}" / f"{split}.h5"
            if not path.exists():
                raise SystemExit(f"Missing required input: {path}")
    return capture("git", "rev-parse", "HEAD"), capture("git", "branch", "--show-current")


def update_registry(commit: str, branch: str, status: str) -> None:
    lines = REGISTRY.read_text().splitlines()
    fields = next(csv.reader([lines[0]]))
    values = {
        "tag": TAG,
        "dataset": "MMECG public",
        "protocol": "loso_v2 complete 11-fold strict LOSO; validation selection then one-time test",
        "category": "final_candidate",
        "publishable": "false",
        "commit_id": commit,
        "branch": branch,
        "role": "Official WSST-driven continuous REM-Net B1 + locked L3 MR-STFT",
        "status": status,
        "reason": (f"Frozen seed-{SEED_BASE} full-fold candidate of the registered three-seed "
                   "audit (42/123/2024); headline claim awaits all three seeds."),
    }
    encoded = io.StringIO()
    writer = csv.DictWriter(encoded, fieldnames=fields, lineterminator="")
    writer.writerow(values)
    replacement = encoded.getvalue()
    replaced = False
    for index, line in enumerate(lines[1:], start=1):
        if next(csv.reader([line]))[0] == TAG:
            lines[index] = replacement
            replaced = True
            break
    if not replaced:
        lines.append(replacement)
    atomic_text(REGISTRY, "\n".join(lines) + "\n")


def run_folds(device: str) -> None:
    for fold in FOLDS:
        fold_dir = OUT / "L3" / f"fold_{fold:02d}"
        result = fold_dir / "test_metrics.json"
        state_path = fold_dir / "run_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        if state and state.get("stage") == "COMPLETE" and result.exists():
            print(f"[skip] fold {fold:02d} already has locked test metrics", flush=True)
            continue
        if state and state.get("stage") == "COMPLETE" and not result.exists():
            raise SystemExit(f"Corrupt COMPLETE fold without metrics: {fold_dir}")
        cmd = [
            sys.executable, "scripts/train_continuous_ecg_pilot.py",
            "--arm", "B1", "--loss_arm", "L3", "--fold", str(fold),
            "--split_root", "loso_v2", "--exp_tag", TAG,
            "--final_test", "--wsst_source", "subject_pool", "--device", device,
            "--seed_base", str(SEED_BASE),
        ]
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


def aggregate() -> None:
    rows = []
    for fold in FOLDS:
        path = OUT / "L3" / f"fold_{fold:02d}" / "test_metrics.json"
        if not path.exists():
            raise SystemExit(f"Cannot aggregate: missing {path}")
        record = json.loads(path.read_text())
        rows.append({"fold": fold, **{k: record["metrics"].get(k) for k in METRICS}})
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_csv(OUT / "test_fold_metrics.csv", list(rows[0]), rows)
    summary = {
        "experiment_id": TAG,
        "aggregation": "fold-macro over 11 held-out test subjects",
        "n_folds": len(rows),
        "metrics": {
            key: {"mean": mean(float(r[key]) for r in rows),
                  "std_population": pstdev(float(r[key]) for r in rows)}
            for key in METRICS
        },
    }
    atomic_json(OUT / "test_fold_macro_summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--seed_base", type=int, default=42, choices=(42, 123, 2024))
    args = parser.parse_args()
    configure(args.seed_base)
    if args.aggregate_only:
        aggregate()
        return
    commit, branch = require_preflight()
    update_registry(commit, branch, "preflight-passed" if args.preflight_only else "running")
    if args.preflight_only:
        print(f"PASS: {TAG} at {commit} on {branch}")
        return
    try:
        run_folds(args.device)
        aggregate()
    except Exception:
        update_registry(commit, branch, "failed; inspect per-fold artifacts")
        raise
    update_registry(commit, branch, "completed; awaiting metric audit and multi-seed decision")


if __name__ == "__main__":
    main()
