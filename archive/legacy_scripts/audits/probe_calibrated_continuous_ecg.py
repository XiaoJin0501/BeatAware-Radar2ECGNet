#!/usr/bin/env python3
"""Record-disjoint subject calibration probe for continuous MMECG ECG.

Uses one validation-subject record for calibration and evaluates every other
record.  Query ECG is never used to fit lag, scale, or offset.  Shape-200 is
deliberately absent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_continuous_ecg_pilot import Fold, waveform_metrics, _set_split_root  # noqa: E402
from src.models.continuous.ce_net import CENet  # noqa: E402

FS = 200
MAX_LAG = 40
BASE = ROOT / "experiments_mmecg/q1_continuous_ecg_b1_lososv2_baseline/B1"
OUT = ROOT / "experiments_mmecg/q1_calibrated_continuous_probe"


def shift_edge(x: np.ndarray, lag: int) -> np.ndarray:
    """Shift predictions on the ECG time axis while retaining 1600 samples."""
    if lag == 0:
        return x.copy()
    y = np.empty_like(x)
    if lag > 0:
        y[:, lag:] = x[:, :-lag]
        y[:, :lag] = x[:, :1]
    else:
        k = -lag
        y[:, :-k] = x[:, k:]
        y[:, -k:] = x[:, -1:]
    return y


def pooled_pcc(p: np.ndarray, y: np.ndarray) -> float:
    p, y = p.reshape(-1), y.reshape(-1)
    p, y = p - p.mean(), y - y.mean()
    d = np.sqrt(np.dot(p, p) * np.dot(y, y))
    return float(np.dot(p, y) / d) if d > 1e-12 else -1.0


def fit_lag(p: np.ndarray, y: np.ndarray) -> int:
    scores = [(pooled_pcc(shift_edge(p, lag), y), lag)
              for lag in range(-MAX_LAG, MAX_LAG + 1)]
    return max(scores, key=lambda z: (z[0], -abs(z[1])))[1]


def fit_affine(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x, t = p.reshape(-1).astype(np.float64), y.reshape(-1).astype(np.float64)
    vx = float(np.var(x))
    a = float(np.mean((x - x.mean()) * (t - t.mean())) / vx) if vx > 1e-12 else 1.0
    return a, float(t.mean() - a * x.mean())


def record_partition(h5_path: Path) -> tuple[str, np.ndarray, np.ndarray, list[str]]:
    with h5py.File(h5_path, "r") as h:
        records = np.asarray([x.decode() if isinstance(x, bytes) else str(x)
                              for x in h["record_id"][:]])
    unique = sorted(set(records.tolist()))
    if len(unique) < 2:
        raise RuntimeError(f"Need >=2 records for record-disjoint calibration: {h5_path}")
    calibration_record = unique[0]
    return calibration_record, records == calibration_record, records != calibration_record, unique


@torch.no_grad()
def predict(fold: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    run = BASE / f"fold_{fold:02d}"
    cfg = json.loads((run / "config.json").read_text())
    scaler = tuple(cfg["scaler_median_iqr"])
    ds = Fold(fold, "val", "B1", scaler)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    model = CENet("B1").to(device)
    ck = torch.load(run / "checkpoints/best_composite.pt", map_location=device, weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    pred, gt = [], []
    for b in loader:
        z, _ = model(b["wsst"].to(device))
        inv = b["inv"].numpy()
        pred.append(z.cpu().numpy()[:, 0] * inv[:, 0:1] + inv[:, 1:2])
        gt.append(b["raw"].numpy()[:, 0])
    return np.concatenate(pred), np.concatenate(gt)


def run_fold(fold: int, device: torch.device) -> dict:
    p, y = predict(fold, device)
    h5 = Path(f"/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_{fold:02d}/val.h5")
    cal_record, cal, query, records = record_partition(h5)
    lag = fit_lag(p[cal], y[cal])
    a0, b0 = fit_affine(p[cal], y[cal])
    shifted = shift_edge(p, lag)
    a1, b1 = fit_affine(shifted[cal], y[cal])
    arms = {
        "zero_shot": p[query],
        "affine": a0 * p[query] + b0,
        "lag": shifted[query],
        "lag_affine": a1 * shifted[query] + b1,
    }
    metrics = {name: waveform_metrics(v, y[query]) for name, v in arms.items()}
    return {
        "fold": fold,
        "calibration_record": cal_record,
        "query_records": [r for r in records if r != cal_record],
        "n_calibration_windows": int(cal.sum()),
        "n_query_windows": int(query.sum()),
        "lag_samples": lag,
        "lag_ms": lag * 1000 / FS,
        "affine_after_lag": {"scale": a1, "offset": b1},
        "metrics": metrics,
        "delta_pcc_lag_affine": metrics["lag_affine"]["pcc"] - metrics["zero_shot"]["pcc"],
    }


def run_session_fold(fold: int, device: torch.device, calibration_seconds: int) -> dict:
    p, y = predict(fold, device)
    h5 = Path(f"/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_{fold:02d}/val.h5")
    with h5py.File(h5, "r") as h:
        records = np.asarray([x.decode() if isinstance(x, bytes) else str(x)
                              for x in h["record_id"][:]])
        start = np.asarray(h["start_idx"][:], np.int64)
        end = np.asarray(h["end_idx"][:], np.int64)
    cutoff = calibration_seconds * FS
    arm_pred = {k: [] for k in ("zero_shot", "affine", "lag", "lag_affine")}
    query_gt, record_rows = [], []
    for record in sorted(set(records.tolist())):
        own = records == record
        cal = own & (end <= cutoff)
        query = own & (start >= cutoff)
        if not cal.any() or not query.any():
            raise RuntimeError(f"{record}: empty calibration or query partition")
        lag = fit_lag(p[cal], y[cal])
        shifted = shift_edge(p, lag)
        a0, b0 = fit_affine(p[cal], y[cal])
        a1, b1 = fit_affine(shifted[cal], y[cal])
        local = {"zero_shot": p[query], "affine": a0 * p[query] + b0,
                 "lag": shifted[query], "lag_affine": a1 * shifted[query] + b1}
        base_pcc = waveform_metrics(local["zero_shot"], y[query])["pcc"]
        cal_pcc = waveform_metrics(local["lag_affine"], y[query])["pcc"]
        record_rows.append({"record_id": record, "n_calibration_windows": int(cal.sum()),
                            "n_query_windows": int(query.sum()), "lag_samples": lag,
                            "lag_ms": lag * 1000 / FS, "delta_pcc": cal_pcc - base_pcc})
        for name, value in local.items():
            arm_pred[name].append(value)
        query_gt.append(y[query])
    target = np.concatenate(query_gt)
    metrics = {name: waveform_metrics(np.concatenate(values), target)
               for name, values in arm_pred.items()}
    return {"fold": fold, "calibration_seconds": calibration_seconds,
            "n_records": len(record_rows), "n_query_windows": len(target),
            "records_improved": sum(r["delta_pcc"] > 0 for r in record_rows),
            "record_results": record_rows, "metrics": metrics,
            "delta_pcc_lag_affine": metrics["lag_affine"]["pcc"] - metrics["zero_shot"]["pcc"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 5, 9])
    ap.add_argument("--split_root", default="loso_v2")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mode", choices=["subject_record", "session_prefix"], default="subject_record")
    ap.add_argument("--calibration_seconds", type=int, default=20)
    args = ap.parse_args()
    _set_split_root(args.split_root)
    runner = (lambda f: run_fold(f, torch.device(args.device))) if args.mode == "subject_record" else (
        lambda f: run_session_fold(f, torch.device(args.device), args.calibration_seconds))
    rows = [runner(f) for f in args.folds]
    deltas = [r["delta_pcc_lag_affine"] for r in rows]
    record_total = sum(r.get("n_records", 0) for r in rows)
    record_improved = sum(r.get("records_improved", 0) for r in rows)
    gate = {"subjects_improved": sum(d > 0 for d in deltas), "subjects_total": len(deltas),
            "mean_delta_pcc": float(np.mean(deltas)),
            "records_improved": record_improved, "records_total": record_total,
            "pass": (all(d > 0 for d in deltas) and float(np.mean(deltas)) >= 0.15
                     and (args.mode != "session_prefix" or record_improved / max(record_total, 1) >= 0.8))}
    spec = ("experiments/specs/q1_calibrated_continuous_probe.json" if args.mode == "subject_record"
            else "experiments/specs/q1_session_calibrated_continuous_probe.json")
    result = {"spec": spec, "mode": args.mode,
              "shape200_computed": False, "folds": rows, "gate": gate}
    out = OUT if args.mode == "subject_record" else ROOT / "experiments_mmecg/q1_session_calibrated_continuous_probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
