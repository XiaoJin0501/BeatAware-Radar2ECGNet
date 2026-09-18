#!/usr/bin/env python3
"""Sequence the matched-protocol baseline benchmark: 5 baselines x 11 `loso_v2` folds.

Mirrors `scripts/run_remnet_continuous_strict_loso.py`. Each fold trains with
validation-only checkpoint selection and then opens its held-out subject exactly
once; a `COMPLETE` fold is terminal and is skipped on re-entry. Per-baseline
fold-macro summaries are written with the same population-std convention as the
REM-Net run, and RR/HR are additionally aggregated under the binding coverage
gate of `docs/METRIC_CONTRACT.md`.

Usage:
  python scripts/run_baseline_continuous_strict_loso.py --device cuda:0
  python scripts/run_baseline_continuous_strict_loso.py --baselines wavegru_net --preflight_only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import median, pstdev, mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.baselines.continuous import BASELINES  # noqa: E402
from src.utils.atomic_state import atomic_csv, atomic_json, atomic_text  # noqa: E402

TAG = "q1_matched_baselines_continuous_strict_loso_seed42"
OUT = ROOT / "experiments_mmecg" / TAG
SPEC = ROOT / "experiments" / "specs" / f"{TAG}.json"
REGISTRY = ROOT / "experiments_mmecg" / "EXPERIMENT_REGISTRY.csv"
FOLDS = list(range(1, 12))
RR_COVERAGE_GATE = 0.80          # METRIC_CONTRACT: binding rate-metric gate

METRICS = ["pcc", "rmse", "nrmse", "mae", "r2", "amplitude_variance_ratio",
           "above_10hz_energy_ratio", "f1_50ms", "f1_100ms", "f1_150ms",
           "rr_error_ms", "hr_error_bpm", "rr_coverage",
           "qrs_region_mae", "stt_region_mae"]
#: Reported as median + IQR, never as a bare mean (METRIC_CONTRACT, rule R2).
HEAVY_TAILED = ["above_10hz_energy_ratio"]


def require_preflight() -> tuple[str, str]:
    """Registered spec, clean worktree, resolvable commit -- before any training."""
    spec = json.loads(SPEC.read_text())
    if spec.get("status") not in {"DATA_READY", "PREFLIGHT_PASSED"}:
        raise SystemExit(f"Spec status {spec.get('status')!r} does not authorise training")
    dirty = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    if dirty:
        raise SystemExit(f"Refusing to train from a dirty worktree:\n{dirty}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    return commit, branch


def update_registry(commit: str, branch: str, status: str) -> None:
    lines = REGISTRY.read_text().splitlines()
    row = (f'{TAG},MMECG public,loso_v2 complete 11-fold strict LOSO; validation selection '
           f'then one-time test,baseline,false,{commit},{branch},'
           f'Matched-protocol continuous baselines (adapted): '
           f'{" / ".join(sorted(BASELINES))},{status},'
           f'"Same split, scaler, evaluator, budget, checkpoint rule and aggregation as '
           f'q1_remnet_continuous_strict_loso_seed42. Adapted reproductions per '
           f'docs/BASELINE_REPRODUCTION_POLICY.md; deviations recorded in each config.json. '
           f'No event head, so event_f1_* is not applicable."')
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith(TAG + ","):
            lines[i], replaced = row, True
            break
    if not replaced:
        lines.append(row)
    atomic_text(REGISTRY, "\n".join(lines) + "\n")


def run_folds(baseline: str, device: str, pretrain_epochs: int) -> None:
    for fold in FOLDS:
        fold_dir = OUT / baseline / f"fold_{fold:02d}"
        result = fold_dir / "test_metrics.json"
        state_path = fold_dir / "run_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        if state and state.get("stage") == "COMPLETE" and result.exists():
            print(f"[skip] {baseline} fold {fold:02d} already has locked test metrics", flush=True)
            continue
        if state and state.get("stage") == "COMPLETE" and not result.exists():
            raise SystemExit(f"Corrupt COMPLETE fold without metrics: {fold_dir}")
        cmd = [sys.executable, "scripts/train_continuous_baseline.py",
               "--baseline", baseline, "--fold", str(fold), "--split_root", "loso_v2",
               "--exp_tag", TAG, "--final_test", "--device", device,
               "--pretrain_epochs", str(pretrain_epochs)]
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


def aggregate(baseline: str) -> dict:
    rows = []
    for fold in FOLDS:
        path = OUT / baseline / f"fold_{fold:02d}" / "test_metrics.json"
        if not path.exists():
            raise SystemExit(f"Cannot aggregate: missing {path}")
        record = json.loads(path.read_text())
        rows.append({"fold": fold, **{k: record["metrics"].get(k) for k in METRICS}})
    atomic_csv(OUT / baseline / "test_fold_metrics.csv", list(rows[0]), rows)

    def col(key):
        return [float(r[key]) for r in rows]

    gated = [r for r in rows if float(r["rr_coverage"]) >= RR_COVERAGE_GATE]
    excluded = [r["fold"] for r in rows if float(r["rr_coverage"]) < RR_COVERAGE_GATE]
    summary = {
        "experiment_id": TAG,
        "baseline": baseline,
        "paper": BASELINES[baseline][2],
        "reproduction_lane": "common-protocol (adapted)",
        "aggregation": "fold-macro over 11 held-out test subjects",
        "std_convention": "population std across folds",
        "n_folds": len(rows),
        "event_metrics_valid": False,
        "metrics": {k: {"mean": mean(col(k)), "std_population": pstdev(col(k))}
                    for k in METRICS},
        "heavy_tailed_metrics": {
            k: {"median": median(col(k)), "min": min(col(k)), "max": max(col(k))}
            for k in HEAVY_TAILED},
        "rate_metrics_gated": {
            "rule": f"rr_coverage >= {RR_COVERAGE_GATE} (docs/METRIC_CONTRACT.md)",
            "n_folds": len(gated),
            "excluded_folds": excluded,
            "rr_error_ms": ({"mean": mean(float(r["rr_error_ms"]) for r in gated),
                             "std_population": pstdev(float(r["rr_error_ms"]) for r in gated)}
                            if gated else None),
            "hr_error_bpm": ({"mean": mean(float(r["hr_error_bpm"]) for r in gated),
                              "std_population": pstdev(float(r["hr_error_bpm"]) for r in gated)}
                             if gated else None)},
    }
    atomic_json(OUT / baseline / "test_fold_macro_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baselines", nargs="*", default=sorted(BASELINES),
                        choices=sorted(BASELINES))
    parser.add_argument("--pretrain_epochs", type=int, default=10)
    parser.add_argument("--preflight_only", action="store_true")
    args = parser.parse_args()

    commit, branch = require_preflight()
    update_registry(commit, branch, "preflight-passed" if args.preflight_only else "running")
    if args.preflight_only:
        print(f"[preflight] spec authorised; commit {commit[:8]} on {branch}")
        return

    summaries = {}
    for baseline in args.baselines:
        print(f"\n=== {baseline} ===", flush=True)
        run_folds(baseline, args.device, args.pretrain_epochs)
        summaries[baseline] = aggregate(baseline)
    atomic_json(OUT / "benchmark_summary.json", {
        "experiment_id": TAG,
        "comparator": "q1_remnet_continuous_strict_loso_seed42 (REM-Net, same protocol)",
        "baselines": summaries})
    update_registry(commit, branch, "completed; awaiting metric audit")
    for name, summary in summaries.items():
        print(f"{name:24s} PCC {summary['metrics']['pcc']['mean']:.4f} "
              f"+/- {summary['metrics']['pcc']['std_population']:.4f}")


if __name__ == "__main__":
    main()
