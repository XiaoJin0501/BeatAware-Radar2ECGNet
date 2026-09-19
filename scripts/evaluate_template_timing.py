#!/usr/bin/env python3
"""Template-timing diagnostics: is the continuous-ECG gap timing or morphology?

Registered analysis `q1_template_timing_diagnostics_seed42`. A fixed beat template,
built from TRAINING-fold labels only, is placed at four sources of R-peak timing
and scored with the frozen evaluator functions, exactly like the models:

  T0  ground-truth R peaks (oracle timing -- an upper bound, never achievable);
  T1  REM-Net's event-head peaks (radar-only);
  T2  peaks detected on REM-Net's reconstructed waveform (radar-only);
  T3  peaks detected on Chen-TCN's reconstructed waveform (radar-only).

Nothing is trained or selected.

Usage:
  python scripts/evaluate_template_timing.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean, pstdev

import h5py
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_anchor_head import peaks_from_heatmap  # noqa: E402
from scripts.train_continuous_ecg_pilot import (  # noqa: E402
    region_errors, smoothing_diagnostics, timing_metrics, waveform_metrics,
)
from src.utils.atomic_state import atomic_csv, atomic_json  # noqa: E402
from src.utils.metrics import detect_rpeaks  # noqa: E402

TAG = "q1_template_timing_diagnostics_seed42"
OUT = ROOT / "experiments_mmecg" / TAG
SPLITS = Path("/home/qhh2237/Datasets/MMECG/processed/loso_v2")
REMNET = ROOT / "experiments_mmecg" / "q1_remnet_continuous_strict_loso_seed42" / "L3"
CHEN = ROOT / "experiments_mmecg" / "q1_matched_baselines_continuous_strict_loso_seed42" / "mmecg_tcn"
FOLDS = list(range(1, 12))
WIN, TEMPLATE_LEN = 1600, 200
METRICS = ("pcc", "rmse", "nrmse", "mae", "r2", "amplitude_variance_ratio",
           "f1_50ms", "f1_100ms", "qrs_region_mae", "stt_region_mae")
VARIANTS = ("remnet_output", "T0_gt", "T1_remnet_event_head", "T2_remnet_waveform",
            "T3_chen_tcn_waveform")


def _clean(peaks) -> np.ndarray:
    r = np.sort(np.unique(np.asarray(peaks, np.int64)))
    return r[(r >= 0) & (r < WIN)]


def train_template(fold: int) -> tuple[np.ndarray, float, float]:
    """Mean R-to-R cycle of the TRAINING fold, its median RR and its mean level."""
    grid = np.linspace(0, 1, TEMPLATE_LEN)
    cycles, rrs = [], []
    with h5py.File(SPLITS / f"fold_{fold:02d}" / "train.h5", "r") as h:
        ecg, peaks = h["ecg"][:, 0], h["rpeak_indices"][:]
    for e, p in zip(ecg, peaks):
        r = _clean(p)
        for a, b in zip(r[:-1], r[1:]):
            if b - a > 10:
                cycles.append(np.interp(grid, np.linspace(0, 1, b - a), e[a:b]))
                rrs.append(b - a)
    return np.mean(cycles, axis=0), float(np.median(rrs)), float(ecg.mean())


def synthesise(peaks: np.ndarray, template: np.ndarray, fallback_rr: float,
               level: float) -> np.ndarray:
    """Place the template between consecutive peaks; pad the ends with whole cycles."""
    out = np.full(WIN, level, dtype=np.float32)
    if len(peaks) == 0:
        return out
    grid = np.linspace(0, 1, TEMPLATE_LEN)

    def cycle(n: int) -> np.ndarray:
        return np.interp(np.linspace(0, 1, n), grid, template)

    rr = float(np.median(np.diff(peaks))) if len(peaks) >= 2 else fallback_rr
    n = max(int(round(rr)), 11)
    for a, b in zip(peaks[:-1], peaks[1:]):
        if b > a:
            out[a:b] = cycle(b - a)
    first, last = int(peaks[0]), int(peaks[-1])
    if first > 0:                                   # a cycle ending on the first peak
        c = cycle(n)
        k = min(first, n)
        out[first - k:first] = c[n - k:]
    if last < WIN:                                  # a cycle starting on the last peak
        c = cycle(n)
        k = min(WIN - last, n)
        out[last:last + k] = c[:k]
        if last + k < WIN:
            out[last + k:] = c[k - 1]
    return out


def score(pred: np.ndarray, gt: np.ndarray, rpeaks) -> dict:
    """The frozen evaluator's metrics, as `evaluate()` computes them."""
    rps = [np.asarray(r, float) for r in rpeaks]
    m = waveform_metrics(pred, gt)
    m.update(smoothing_diagnostics(pred, gt))
    m.update(timing_metrics(pred, np.zeros_like(pred), rps))
    m.update(region_errors(pred, gt, rps))
    return {k: m[k] for k in METRICS}


def main() -> None:
    rows = []
    per = {v: [] for v in VARIANTS}
    failures = {v: 0 for v in VARIANTS}
    for fold in FOLDS:
        rem = np.load(REMNET / f"fold_{fold:02d}" / "test_predictions.npz", allow_pickle=True)
        chen = np.load(CHEN / f"fold_{fold:02d}" / "test_predictions.npz", allow_pickle=True)
        gt, rpeaks = rem["ground_truth"], rem["rpeaks"]
        if not np.array_equal(gt, chen["ground_truth"]):
            raise SystemExit(f"fold {fold}: REM-Net and Chen-TCN test windows differ")
        template, fallback_rr, level = train_template(fold)

        sources = {
            "T0_gt": [_clean(r) for r in rpeaks],
            "T1_remnet_event_head": [_clean(peaks_from_heatmap(p[None]))
                                     for p in rem["event_probability"]],
            "T2_remnet_waveform": [_clean(detect_rpeaks(p)) for p in rem["prediction"]],
            "T3_chen_tcn_waveform": [_clean(detect_rpeaks(p)) for p in chen["prediction"]],
        }
        results = {"remnet_output": score(rem["prediction"], gt, rpeaks)}
        for name, peak_sets in sources.items():
            failures[name] += sum(len(p) == 0 for p in peak_sets)
            synth = np.stack([synthesise(p, template, fallback_rr, level) for p in peak_sets])
            results[name] = score(synth, gt, rpeaks)
        for name, r in results.items():
            per[name].append(r)
            rows.append({"variant": name, "fold": fold, **r})
        print(f"fold {fold:2d}: " + "  ".join(f"{n.split('_')[0]} {r['pcc']:+.3f}"
                                              for n, r in results.items()), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    atomic_csv(OUT / "template_timing_fold_metrics.csv", list(rows[0]), rows)
    summary = {v: {k: {"mean": mean(r[k] for r in rs), "std_population": pstdev(r[k] for r in rs)}
                   for k in METRICS} for v, rs in per.items()}
    ref = per["remnet_output"]
    comparisons = {}
    for v in VARIANTS[1:]:
        comparisons[v] = {}
        for k in ("pcc", "r2", "nrmse", "f1_100ms", "qrs_region_mae"):
            a = np.array([r[k] for r in per[v]]); b = np.array([r[k] for r in ref])
            comparisons[v][k] = {"variant_minus_remnet": float((a - b).mean()),
                                 "wilcoxon_p": float(wilcoxon(a, b).pvalue),
                                 "variant_higher_folds": int(((a - b) > 0).sum())}
    atomic_json(OUT / "template_timing_summary.json", {
        "experiment_id": TAG,
        "template": "training-fold mean R-to-R cycle, raw per-record z-scored ECG",
        "aggregation": "window-macro within fold, fold-macro over 11 folds, population std",
        "windows_without_peaks": failures,
        "variants": summary, "versus_remnet_output": comparisons})

    print(f"\n{'variant':22s}" + "".join(f"{k:>16s}" for k in ("pcc", "r2", "nrmse", "f1_100ms",
                                                              "qrs_region_mae", "var_ratio")))
    for v, s in summary.items():
        print(f"{v:22s}" + "".join(
            f"  {s[k]['mean']:+.3f}±{s[k]['std_population']:.3f}"
            for k in ("pcc", "r2", "nrmse", "f1_100ms", "qrs_region_mae",
                      "amplitude_variance_ratio")))
    print(f"\nwindows with no peaks: {failures}")
    print("\nversus REM-Net's own continuous output (paired Wilcoxon, 11 folds):")
    for v, c in comparisons.items():
        print(f"  {v:22s} " + " | ".join(
            f"{k}: Δ{d['variant_minus_remnet']:+.3f} p={d['wilcoxon_p']:.3f} "
            f"hi {d['variant_higher_folds']}/11" for k, d in c.items()))


if __name__ == "__main__":
    main()
