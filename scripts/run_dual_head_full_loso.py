#!/usr/bin/env python3
"""Complete the locked dual-head REM-Net 11-fold seed-42 benchmark."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_dual_head_interaction as core  # noqa: E402
from src.utils.atomic_state import atomic_json  # noqa: E402

TAG = "q1_dual_head_full_loso_seed42"
SPEC = ROOT / "experiments/specs/q1_dual_head_full_loso_seed42.json"
OUT = Path(os.environ.get("REM_DUAL_HEAD_FULL_OUT", ROOT / "experiments_mmecg" / TAG))
SOURCE = Path(os.environ.get(
    "REM_DUAL_HEAD_GATE_OUT",
    ROOT / "experiments_mmecg/q1_dual_head_interaction_gate"))
FOLDS = tuple(range(1, 12))
REUSED = (2, 6, 10)


def preflight() -> None:
    spec = json.loads(SPEC.read_text())
    if not spec["status"].startswith("PREFLIGHT_PASSED"):
        raise SystemExit(f"spec not launchable: {spec['status']}")
    commit = spec.get("implementation_commit")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()
    if not commit or subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head], cwd=ROOT).returncode:
        raise SystemExit("implementation commit is missing or not an ancestor")
    for fold in FOLDS:
        for split in ("train", "val", "test"):
            path = Path(f"/home/qhh2237/Datasets/MMECG/processed/loso_v2/"
                        f"fold_{fold:02d}/{split}.h5")
            if not path.exists(): raise SystemExit(f"missing {path}")
    for fold in REUSED:
        src = SOURCE / "A1" / f"fold_{fold:02d}"
        if json.loads((src / "run_state.json").read_text())["stage"] != "COMPLETE":
            raise SystemExit(f"reuse source incomplete: {src}")


def establish_reuse() -> None:
    arm = OUT / "A1"; arm.mkdir(parents=True, exist_ok=True)
    for fold in REUSED:
        dst = arm / f"fold_{fold:02d}"
        src = (SOURCE / "A1" / f"fold_{fold:02d}").resolve()
        if dst.is_symlink():
            if dst.resolve() != src: raise RuntimeError(f"wrong reuse link: {dst}")
        elif dst.exists():
            raise RuntimeError(f"refusing to replace existing path: {dst}")
        else:
            dst.symlink_to(src, target_is_directory=True)


def aggregate() -> dict:
    folds = {}
    for fold in FOLDS:
        folds[str(fold)] = json.loads(
            (OUT / "A1" / f"fold_{fold:02d}/test_metrics.json").read_text())
    shape = {}
    for metric in ("pcc", "rmse", "r2"):
        values = np.asarray([folds[str(f)]["shape"][metric] for f in FOLDS])
        shape[metric] = {"mean": float(values.mean()), "std_population": float(values.std()),
                         "per_fold": values.tolist()}
    event = {}
    for p in core.FIDUCIALS:
        for metric in ("medae_ms", "coverage", "mdr_150ms", "f1_100ms"):
            raw = [folds[str(f)]["structured"][p][metric] for f in FOLDS]
            values = np.asarray([np.nan if x is None else x for x in raw], dtype=float)
            valid = values[np.isfinite(values)]
            event[f"{p}_{metric}"] = {
                "mean": float(valid.mean()) if valid.size else None,
                "std_population": float(valid.std()) if valid.size else None,
                "median": float(np.median(valid)) if valid.size else None,
                "n_valid_folds": int(valid.size),
                "n_missing_folds": int(len(values) - valid.size),
                "per_fold": [None if not np.isfinite(x) else float(x) for x in values]}
    order = [folds[str(f)]["structured"]["order_violation_rate"] for f in FOLDS]
    result = {"experiment_id": TAG, "seed": 42, "folds": folds,
              "shape_fold_macro": shape, "structured_event_fold_macro": event,
              "order_violation_rate_per_fold": order,
              "reuse_provenance": {str(f): str((SOURCE/"A1"/f"fold_{f:02d}").resolve())
                                   for f in REUSED}}
    atomic_json(OUT / "summary.json", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fold", type=int, choices=FOLDS)
    p.add_argument("--preflight_only", action="store_true")
    p.add_argument("--aggregate_only", action="store_true")
    args = p.parse_args()
    preflight(); establish_reuse()
    if args.preflight_only:
        print("PASS", TAG); return
    if args.aggregate_only:
        print(json.dumps(aggregate()["shape_fold_macro"], indent=2)); return
    core.OUT = OUT; core.SEED = 42
    folds = (args.fold,) if args.fold else FOLDS
    for fold in folds:
        core.train(fold, "A1", core.torch.device(args.device))
    if all((OUT/"A1"/f"fold_{f:02d}/test_metrics.json").exists() for f in FOLDS):
        print(json.dumps(aggregate()["shape_fold_macro"], indent=2))


if __name__ == "__main__": main()
