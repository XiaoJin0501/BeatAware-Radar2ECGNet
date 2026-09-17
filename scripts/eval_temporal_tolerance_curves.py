#!/usr/bin/env python3
"""GT-assisted temporal-tolerance curves for frozen continuous B1 outputs.

This is a validation-only diagnostic. Max-lag NCC is an oracle score because
the reference ECG chooses the lag; it is never deployable performance.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_calibration_conditioned_rem import freeze_predictions  # noqa: E402
from scripts.train_continuous_ecg_pilot import waveform_metrics  # noqa: E402

FS = 200
MAX_LAG_MS = 100
MAX_LAG = MAX_LAG_MS * FS // 1000
TOLERANCES_MS = list(range(0, MAX_LAG_MS + 1, 10))
OUT = ROOT / "experiments_mmecg/q1_temporal_tolerance_curve_audit"


def bandpass_zscore(x: np.ndarray, fs: int = FS) -> np.ndarray:
    """Zero-phase 0.5-40 Hz bandpass, then independent window z-score."""
    sos = butter(4, (0.5, 40.0), btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, np.asarray(x, np.float64), axis=-1)
    return (y - y.mean(-1, keepdims=True)) / np.maximum(y.std(-1, keepdims=True), 1e-8)


def lag_scores(pred: np.ndarray, target: np.ndarray, max_lag: int = MAX_LAG) -> tuple[np.ndarray, np.ndarray]:
    """Per-window PCC at every lag using one fixed-length common support."""
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError(f"expected equal (N,T) arrays, got {pred.shape} and {target.shape}")
    n_time = pred.shape[1]
    if n_time <= 2 * max_lag:
        raise ValueError("window shorter than twice max_lag")
    reference = target[:, max_lag:n_time - max_lag]
    reference = reference - reference.mean(-1, keepdims=True)
    reference_norm = np.sqrt(np.sum(reference * reference, axis=-1))
    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    scores = np.empty((len(pred), len(lags)), np.float64)
    for j, lag in enumerate(lags):
        shifted = pred[:, max_lag + lag:n_time - max_lag + lag]
        shifted = shifted - shifted.mean(-1, keepdims=True)
        denominator = reference_norm * np.sqrt(np.sum(shifted * shifted, axis=-1))
        numerator = np.sum(reference * shifted, axis=-1)
        scores[:, j] = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 1e-12)
    return lags, scores


def tolerance_curve(lags: np.ndarray, scores: np.ndarray) -> tuple[list[dict], np.ndarray]:
    rows = []
    for tolerance_ms in TOLERANCES_MS:
        tolerance_samples = int(round(tolerance_ms * FS / 1000))
        allowed = np.abs(lags) <= tolerance_samples
        best = np.nanmax(scores[:, allowed], axis=1)
        rows.append({"tolerance_ms": tolerance_ms, "mean_ncc": float(np.nanmean(best)),
                     "median_ncc": float(np.nanmedian(best)), "std_ncc": float(np.nanstd(best)),
                     "n_windows": int(np.isfinite(best).sum())})
    best_idx = np.nanargmax(scores, axis=1)
    return rows, lags[best_idx]


def evaluate_fold(fold: int, device: torch.device) -> dict:
    data = freeze_predictions(fold, "val", device)
    pred = data["pred"][:, 0].numpy()
    target = data["gt"][:, 0].numpy()
    raw = waveform_metrics(pred, target)
    pred_f, target_f = bandpass_zscore(pred), bandpass_zscore(target)
    lags, scores = lag_scores(pred_f, target_f)
    curve, best_lags = tolerance_curve(lags, scores)
    return {"fold": fold, "n_windows": len(pred), "raw_full_window_pcc_at_0": raw["pcc"],
            "filtered_fixed_support_pcc_at_0": curve[0]["mean_ncc"], "curve": curve,
            "best_lags_samples_at_100ms": best_lags.tolist(),
            "best_lags_ms_at_100ms": (best_lags * 1000 / FS).tolist()}


def plot_results(folds: list[dict], macro: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    for row in folds:
        axes[0].plot(TOLERANCES_MS, [x["mean_ncc"] for x in row["curve"]], marker="o", alpha=.65,
                     label=f"fold {row['fold']}")
    axes[0].plot(TOLERANCES_MS, [x["mean_ncc"] for x in macro], color="black", marker="o",
                 linewidth=2.4, label="fold-macro")
    axes[0].set(xlabel="Maximum oracle lag (ms)", ylabel="Filtered max-lag NCC",
                title="Temporal-tolerance curve")
    axes[0].grid(alpha=.25); axes[0].legend(frameon=False)
    lag_values = np.concatenate([np.asarray(x["best_lags_ms_at_100ms"]) for x in folds])
    axes[1].hist(lag_values, bins=np.arange(-102.5, 107.5, 5), color="#2878b5", alpha=.85)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set(xlabel="Oracle-selected lag at +/-100 ms (ms)", ylabel="Windows",
                title="Lag distribution")
    axes[1].grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "temporal_tolerance_curves.png", dpi=220)
    fig.savefig(out / "temporal_tolerance_curves.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default="1,5,9")
    parser.add_argument("--split", default="val", choices=["val"], help="test is deliberately unavailable")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    folds = [int(x) for x in args.folds.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    results = [evaluate_fold(fold, torch.device(args.device)) for fold in folds]
    macro = []
    for i, tolerance_ms in enumerate(TOLERANCES_MS):
        values = [row["curve"][i]["mean_ncc"] for row in results]
        macro.append({"tolerance_ms": tolerance_ms, "mean_ncc": float(np.mean(values)),
                      "std_across_folds": float(np.std(values)), "n_folds": len(values)})
    payload = {"experiment_id": "q1_temporal_tolerance_curve_audit", "tier": "probe",
               "publishable": False, "split": "validation", "test_opened": False,
               "filter": "Butterworth order 4, zero-phase, 0.5-40 Hz; per-window z-score",
               "support": f"fixed target samples {MAX_LAG}:{1600-MAX_LAG} for every lag",
               "oracle_warning": "GT selects lag; diagnostic only, not deployable performance",
               "folds": results, "fold_macro_curve": macro}
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (OUT / "curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "fold", "tolerance_ms", "mean_ncc", "spread"])
        writer.writeheader()
        for row in results:
            for point in row["curve"]:
                writer.writerow({"scope": "fold", "fold": row["fold"], "tolerance_ms": point["tolerance_ms"],
                                 "mean_ncc": point["mean_ncc"], "spread": point["std_ncc"]})
        for point in macro:
            writer.writerow({"scope": "fold_macro", "fold": "", "tolerance_ms": point["tolerance_ms"],
                             "mean_ncc": point["mean_ncc"], "spread": point["std_across_folds"]})
    plot_results(results, macro, OUT)
    print(json.dumps({"raw_pcc_at_0_fold_macro": float(np.mean([x["raw_full_window_pcc_at_0"] for x in results])),
                      "filtered_curve": macro}, indent=2))


if __name__ == "__main__":
    main()
