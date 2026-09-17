#!/usr/bin/env python3
"""Orchestrate the registered 11-fold continuous REM-Net final candidate.

This entry point does not alter the model or select from test results. Each
fold trains on train, selects one checkpoint on validation, and only then opens
the held-out test subject once. Existing completed folds are resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
TAG = "q1_remnet_continuous_strict_loso_seed42"
SPEC = ROOT / "experiments" / "specs" / f"{TAG}.json"
REGISTRY = ROOT / "experiments_mmecg" / "EXPERIMENT_REGISTRY.csv"
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
    if spec.get("status", "").startswith("PREPARED") is False:
        raise SystemExit(f"Unexpected spec state: {spec.get('status')}")
    dirty = capture("git", "status", "--short", "--untracked-files=no")
    if dirty:
        raise SystemExit("Refusing formal launch with uncommitted tracked files:\n" + dirty)
    split_root = Path("/home/qhh2237/Datasets/MMECG/processed/loso_v2")
    for fold in FOLDS:
        for split in ("train", "val", "test"):
            for suffix in (".h5", "_wsst8s.h5"):
                path = split_root / f"fold_{fold:02d}" / f"{split}{suffix}"
                if not path.exists():
                    raise SystemExit(f"Missing required input: {path}")
    return capture("git", "rev-parse", "HEAD"), capture("git", "branch", "--show-current")


def update_registry(commit: str, branch: str, status: str) -> None:
    with REGISTRY.open(newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0])
    row = next((r for r in rows if r["tag"] == TAG), None)
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
        "reason": "Frozen seed-42 full-fold candidate; headline claim awaits metric audit and multi-seed confirmation.",
    }
    if row is None:
        rows.append(values)
    else:
        row.update(values)
    with REGISTRY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_folds(device: str) -> None:
    for fold in FOLDS:
        result = OUT / "L3" / f"fold_{fold:02d}" / "test_metrics.json"
        if result.exists():
            print(f"[skip] fold {fold:02d} already has locked test metrics", flush=True)
            continue
        cmd = [
            sys.executable, "scripts/train_continuous_ecg_pilot.py",
            "--arm", "B1", "--loss_arm", "L3", "--fold", str(fold),
            "--split_root", "loso_v2", "--exp_tag", TAG,
            "--final_test", "--device", device,
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
    with (OUT / "test_fold_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
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
    (OUT / "test_fold_macro_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()
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
