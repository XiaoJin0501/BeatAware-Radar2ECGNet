#!/usr/bin/env python3
"""Evaluate paired-ECG 60-second priming calibration on frozen LOSO outputs.

This is a separate clinically calibrated track, not strict zero-shot LOSO.
For each held-out subject, only the first record's source interval [0, 60 s]
estimates one fixed lag and one multiplicative standard-deviation scale. Windows
straddling 60 s are excluded so calibration and query share no source samples.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_continuous_ecg_pilot import timing_metrics, waveform_metrics  # noqa: E402
from src.utils.atomic_state import atomic_json  # noqa: E402

FS = 200
BOUNDARY = 60 * FS
MAX_LAG = int(round(0.150 * FS))
SOURCE = ROOT / "experiments_mmecg/q1_remnet_continuous_strict_loso_seed42/L3"
SPLITS = Path("/home/qhh2237/Datasets/MMECG/processed/loso_v2")
OUTPUT = ROOT / "experiments_mmecg/q1_remnet_60s_calibration_summary.json"


def text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def chronological_masks(record_id: np.ndarray, start: np.ndarray, end: np.ndarray):
    """Strict non-overlap split around 60 s of the first chronological record."""
    first_record = text(record_id[0])
    is_first = np.asarray([text(value) == first_record for value in record_id])
    calibration = is_first & (end <= BOUNDARY)
    evaluation = (~is_first) | (start >= BOUNDARY)
    excluded = ~(calibration | evaluation)
    if np.any(calibration & evaluation):
        raise RuntimeError("calibration/evaluation window overlap")
    if int(end[calibration].max()) > int(start[evaluation & is_first].min()):
        raise RuntimeError("calibration/evaluation source samples overlap")
    return calibration, evaluation, excluded, first_record


def shifted(prediction: np.ndarray, lag: int) -> np.ndarray:
    """Return corrected[t] = prediction[t+lag], with edge-value padding."""
    if prediction.ndim != 2:
        raise ValueError("prediction must be [window,time]")
    result = np.empty_like(prediction)
    if lag > 0:
        result[:, :-lag] = prediction[:, lag:]
        result[:, -lag:] = prediction[:, -1:]
    elif lag < 0:
        amount = -lag
        result[:, amount:] = prediction[:, :-amount]
        result[:, :amount] = prediction[:, :1]
    else:
        result[:] = prediction
    return result


def pcc_rows(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = prediction - prediction.mean(1, keepdims=True)
    y = target - target.mean(1, keepdims=True)
    denominator = np.linalg.norm(p, axis=1) * np.linalg.norm(y, axis=1)
    return (p * y).sum(1) / np.maximum(denominator, 1e-12)


def estimate_calibration(prediction: np.ndarray, target: np.ndarray) -> tuple[int, float, list[float]]:
    """Estimate lag and scale using calibration arrays only."""
    lag_scores = []
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        lag_scores.append(float(np.nanmean(pcc_rows(shifted(prediction, lag), target))))
    best_lag = int(np.argmax(lag_scores)) - MAX_LAG
    aligned = shifted(prediction, best_lag)
    alpha = float(np.std(target) / max(float(np.std(aligned)), 1e-12))
    return best_lag, alpha, lag_scores


def metrics(prediction: np.ndarray, target: np.ndarray, rpeaks: np.ndarray) -> dict[str, float]:
    result = waveform_metrics(prediction, target)
    dummy_event = np.zeros_like(prediction)
    result.update({
        "f1_100ms": timing_metrics(prediction, dummy_event, rpeaks)["f1_100ms"],
    })
    return {key: float(result[key]) for key in ("pcc", "r2", "nrmse", "f1_100ms")}


def aggregate(rows: list[dict], metric: str, condition: str) -> dict[str, float]:
    values = np.asarray([row[condition][metric] for row in rows], dtype=float)
    return {"mean": float(values.mean()), "std_population": float(values.std(ddof=0))}


def main() -> None:
    folds = []
    for fold in range(1, 12):
        prediction_path = SOURCE / f"fold_{fold:02d}" / "test_predictions.npz"
        split_path = SPLITS / f"fold_{fold:02d}" / "test.h5"
        arrays = np.load(prediction_path, allow_pickle=True)
        prediction = np.asarray(arrays["prediction"], dtype=np.float32)
        target = np.asarray(arrays["ground_truth"], dtype=np.float32)
        rpeaks = arrays["rpeaks"]
        with h5py.File(split_path, "r") as handle:
            record_id = handle["record_id"][:]
            start = np.asarray(handle["start_idx"][:], dtype=np.int64)
            end = np.asarray(handle["end_idx"][:], dtype=np.int64)
        if not (len(prediction) == len(record_id) == len(target) == len(rpeaks)):
            raise RuntimeError(f"fold {fold}: prediction/metadata length mismatch")

        calibration, evaluation, excluded, first_record = chronological_masks(record_id, start, end)
        lag, alpha, lag_scores = estimate_calibration(
            prediction[calibration], target[calibration]
        )
        corrected = alpha * shifted(prediction[evaluation], lag)
        zero = metrics(prediction[evaluation], target[evaluation], rpeaks[evaluation])
        calibrated = metrics(corrected, target[evaluation], rpeaks[evaluation])
        folds.append({
            "fold": fold,
            "first_record": first_record,
            "n_total": int(len(prediction)),
            "n_calibration": int(calibration.sum()),
            "n_boundary_excluded": int(excluded.sum()),
            "n_evaluation": int(evaluation.sum()),
            "calibration_source_interval_seconds": [0, 60],
            "lag_samples": lag,
            "lag_ms": lag * 1000.0 / FS,
            "alpha": alpha,
            "calibration_mean_pcc_at_selected_lag": lag_scores[lag + MAX_LAG],
            "zero_calib": zero,
            "calibrated_60s": calibrated,
            "delta": {key: calibrated[key] - zero[key] for key in zero},
        })

    macro = {}
    for metric in ("pcc", "r2", "nrmse", "f1_100ms"):
        macro[metric] = {
            "zero_calib": aggregate(folds, metric, "zero_calib"),
            "calibrated_60s": aggregate(folds, metric, "calibrated_60s"),
        }
        macro[metric]["delta_mean"] = (
            macro[metric]["calibrated_60s"]["mean"] - macro[metric]["zero_calib"]["mean"]
        )

    output = {
        "experiment_id": "q1_remnet_60s_priming_calibration",
        "category": "paired_calibration_posthoc",
        "publishable": False,
        "source_experiment": "q1_remnet_continuous_strict_loso_seed42",
        "protocol": {
            "calibration": "first record windows ending at or before 60 s",
            "boundary": "windows straddling 60 s excluded",
            "evaluation": "same-record windows starting at or after 60 s plus all later records",
            "overlap": "no source-sample overlap between calibration and evaluation",
            "lag_grid_ms": [-150, 150, 5],
            "lag_objective": "mean per-window calibration PCC",
            "scale": "global calibration std(GT)/std(prediction after lag)",
            "warning": "uses paired held-out-subject ECG; not strict zero-shot LOSO",
        },
        "folds": folds,
        "fold_macro": macro,
    }
    atomic_json(OUTPUT, output)

    print("| Fold | Calib/Eval | Lag (ms) | Alpha | PCC zero | PCC 60s | Delta PCC |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for row in folds:
        print(
            f"| {row['fold']} | {row['n_calibration']}/{row['n_evaluation']} | "
            f"{row['lag_ms']:+.0f} | {row['alpha']:.3f} | "
            f"{row['zero_calib']['pcc']:.3f} | {row['calibrated_60s']['pcc']:.3f} | "
            f"{row['delta']['pcc']:+.3f} |"
        )
    z = macro["pcc"]["zero_calib"]
    c = macro["pcc"]["calibrated_60s"]
    print(
        f"\nFold-macro PCC: {z['mean']:.3f} +/- {z['std_population']:.3f} -> "
        f"{c['mean']:.3f} +/- {c['std_population']:.3f} "
        f"(delta {macro['pcc']['delta_mean']:+.3f})"
    )
    print(f"Result: {OUTPUT}")


if __name__ == "__main__":
    main()
