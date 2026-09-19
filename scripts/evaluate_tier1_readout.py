#!/usr/bin/env python3
"""Tier-1 (Shape-200) morphology readout of the frozen continuous benchmark.

Registered analysis `q1_tier1_readout_from_continuous_seed42`. For every test
window of REM-Net and the five matched baselines, the centre cardiac cycle is cut
with the ground-truth R peaks, resampled to 200 samples and min-max normalised on
both sides -- the Tier-1 known-cycle definition used by
`scripts/train_continuous_ecg_pilot.shape200_pcc`. Nothing is trained, tuned or
selected: the inputs are the test predictions each run already froze.

Usage:
  python scripts/evaluate_tier1_readout.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_continuous_ecg_pilot import _centre_cycle, shape200_pcc  # noqa: E402
from src.utils.atomic_state import atomic_csv, atomic_json  # noqa: E402

TAG = "q1_tier1_readout_from_continuous_seed42"
OUT = ROOT / "experiments_mmecg" / TAG
SHAPE_LEN = 200
FOLDS = list(range(1, 12))
MODELS = {
    "rem_net": ROOT / "experiments_mmecg" / "q1_remnet_continuous_strict_loso_seed42" / "L3",
    **{name: ROOT / "experiments_mmecg" / "q1_matched_baselines_continuous_strict_loso_seed42" / name
       for name in ("mmecg_tcn", "radarode_mtl", "wavegru_net", "attention_dcgan",
                    "cross_modal_conformer")},
}


def _norm01(z: np.ndarray) -> np.ndarray:
    return (z - z.min()) / max(z.max() - z.min(), 1e-8)


def fold_readout(npz_path: Path) -> dict:
    """Window-macro Shape-200 PCC / RMSE / MAE for one fold's frozen predictions."""
    data = np.load(npz_path, allow_pickle=True)
    pred, gt, rpeaks = data["prediction"], data["ground_truth"], data["rpeaks"]
    grid = np.linspace(0, 1, SHAPE_LEN)
    pcc, rmse, mae = [], [], []
    for p, y, r in zip(pred, gt, rpeaks):
        bounds = _centre_cycle(np.asarray(r).astype(int))
        if bounds is None:
            continue
        a, b = bounds
        src = np.linspace(0, 1, b - a)
        ps, ys = _norm01(np.interp(grid, src, p[a:b])), _norm01(np.interp(grid, src, y[a:b]))
        pc, yc = ps - ps.mean(), ys - ys.mean()
        den = np.sqrt((pc ** 2).sum() * (yc ** 2).sum())
        if den <= 1e-12:
            continue
        pcc.append(float((pc * yc).sum() / den))
        rmse.append(float(np.sqrt(((ps - ys) ** 2).mean())))
        mae.append(float(np.abs(ps - ys).mean()))
    reference = shape200_pcc(pred, gt, rpeaks)
    if not np.isclose(np.mean(pcc), reference, atol=1e-9):
        raise SystemExit(f"Tier-1 PCC disagrees with shape200_pcc on {npz_path}: "
                         f"{np.mean(pcc):.6f} vs {reference:.6f}")
    return {"shape200_pcc": float(np.mean(pcc)), "shape200_rmse": float(np.mean(rmse)),
            "shape200_mae": float(np.mean(mae)), "n_windows": int(len(pred)),
            "n_valid_cycles": int(len(pcc)), "cycle_coverage": len(pcc) / max(len(pred), 1)}


def main() -> None:
    rows, per_model = [], {}
    for model, root in MODELS.items():
        per_model[model] = []
        for fold in FOLDS:
            path = root / f"fold_{fold:02d}" / "test_predictions.npz"
            if not path.exists():
                raise SystemExit(f"Missing frozen predictions: {path}")
            r = fold_readout(path)
            per_model[model].append(r)
            rows.append({"model": model, "fold": fold, **r})
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_csv(OUT / "tier1_fold_metrics.csv", list(rows[0]), rows)

    keys = ("shape200_pcc", "shape200_rmse", "shape200_mae", "cycle_coverage")
    summary = {m: {k: {"mean": mean(r[k] for r in rs), "std_population": pstdev(r[k] for r in rs)}
                   for k in keys} for m, rs in per_model.items()}

    ref = per_model["rem_net"]
    comparisons = {}
    for model, rs in per_model.items():
        if model == "rem_net":
            continue
        comparisons[model] = {}
        for k, better in (("shape200_pcc", 1), ("shape200_rmse", -1), ("shape200_mae", -1)):
            a = np.array([r[k] for r in ref]); b = np.array([r[k] for r in rs])
            d = a - b
            comparisons[model][k] = {
                "rem_minus_baseline": float(d.mean()),
                "wilcoxon_p": float(wilcoxon(a, b).pvalue),
                "rem_net_wins": int(((better * d) > 0).sum()),
            }
    atomic_json(OUT / "tier1_summary.json", {
        "experiment_id": TAG,
        "definition": "known-cycle Shape-200 morphology of frozen continuous test predictions",
        "aggregation": "window-macro within fold, fold-macro across 11 test subjects",
        "std_convention": "population std across folds",
        "statistics": "paired two-sided Wilcoxon on 11 fold values; descriptive, uncorrected",
        "models": summary, "rem_net_vs_baseline": comparisons})

    print(f"{'model':24s} {'Shape200 PCC':>16s} {'RMSE':>14s} {'MAE':>14s} {'coverage':>9s}")
    for m, s in sorted(summary.items(), key=lambda kv: -kv[1]["shape200_pcc"]["mean"]):
        print(f"{m:24s} {s['shape200_pcc']['mean']:.4f} ± {s['shape200_pcc']['std_population']:.4f}"
              f" {s['shape200_rmse']['mean']:.4f} ± {s['shape200_rmse']['std_population']:.3f}"
              f" {s['shape200_mae']['mean']:.4f} ± {s['shape200_mae']['std_population']:.3f}"
              f" {s['cycle_coverage']['mean']:9.3f}")
    print("\nREM-Net vs baseline (paired Wilcoxon, 11 folds):")
    for m, c in comparisons.items():
        print(f"  {m:24s} " + " | ".join(
            f"{k.split('_')[1]}: Δ{v['rem_minus_baseline']:+.4f} p={v['wilcoxon_p']:.3f} "
            f"win {v['rem_net_wins']}/11" for k, v in c.items()))


if __name__ == "__main__":
    main()
