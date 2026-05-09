"""
analyze_mmecg_input_lag.py — estimate radar-input vs ECG timing lag.

This diagnostic does not load any model. It reads processed MMECG H5 files and
compares each RCG window against the ECG R-peak train within a bounded lag
window. The goal is to determine whether subject/scene timing drift is already
present in the input data or mainly introduced by the reconstruction model.

Outputs:
  experiments_mmecg/input_lag_<protocol>[_foldXX]/
    input_lag_metrics.csv
    input_lag_summary_by_subject_scene.csv
    input_lag_summary_by_subject.csv
    input_lag_summary_by_scene.csv
    input_lag_global_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.mmecg_config import MMECGConfig


FS = 200
WIN_LEN = 1600
STATE_NAMES = {b"NB": "NB", b"IB": "IB", b"SP": "SP", b"PE": "PE"}


def _decode_state(x) -> str:
    if isinstance(x, bytes):
        return STATE_NAMES.get(x, x.decode())
    return str(x)


def _gaussian_mask(indices, length: int = WIN_LEN, sigma: float = 5.0) -> np.ndarray:
    mask = np.zeros(length, dtype=np.float32)
    for idx in np.asarray(indices, dtype=np.int32):
        if idx < 0:
            continue
        lo = max(0, int(idx) - int(3 * sigma))
        hi = min(length, int(idx) + int(3 * sigma) + 1)
        t = np.arange(lo, hi)
        mask[lo:hi] += np.exp(-0.5 * ((t - int(idx)) / sigma) ** 2)
    return np.clip(mask, 0.0, 1.0)


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)


def _bandpass_rcg(rcg: np.ndarray, fs: float = FS) -> np.ndarray:
    b, a = butter(3, [0.8, 8.0], btype="band", fs=fs)
    return filtfilt(b, a, rcg, axis=-1).astype(np.float32)


def _shifted_views(a: np.ndarray, b: np.ndarray, lag: int):
    """
    Positive lag means a is compared as a[lag:] vs b[:-lag].
    For this script, a=RCG and b=ECG R-peak train.
    """
    if lag > 0:
        return a[..., lag:], b[:-lag]
    if lag < 0:
        return a[..., :lag], b[-lag:]
    return a, b


def _corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-channel Pearson correlation. a: (C,L), b: (L,)."""
    ac = a - a.mean(axis=-1, keepdims=True)
    bc = b - b.mean()
    num = (ac * bc[None, :]).sum(axis=-1)
    den = np.sqrt((ac * ac).sum(axis=-1) * np.sum(bc * bc)) + 1e-8
    return num / den


def _best_input_lag(
    rcg: np.ndarray,
    rpeak_train: np.ndarray,
    max_lag_samples: int,
) -> dict:
    """
    Estimate lag using two views:
      - best_channel: pick the RCG channel with strongest absolute correlation.
      - mean_abs: use mean absolute RCG amplitude across channels.
    """
    rcg = _zscore(_bandpass_rcg(rcg))
    rcg_abs_mean = _zscore(np.mean(np.abs(rcg), axis=0, keepdims=True))

    best = {
        "best_channel_lag_samples": 0,
        "best_channel_corr_abs": -np.inf,
        "best_channel_corr_signed": 0.0,
        "best_channel_idx": -1,
        "mean_abs_lag_samples": 0,
        "mean_abs_corr_abs": -np.inf,
        "mean_abs_corr_signed": 0.0,
    }

    for lag in range(-max_lag_samples, max_lag_samples + 1):
        rcg_view, peak_view = _shifted_views(rcg, rpeak_train, lag)
        if rcg_view.shape[-1] < WIN_LEN * 0.75:
            continue
        corrs = _corr_rows(rcg_view, peak_view)
        idx = int(np.argmax(np.abs(corrs)))
        corr = float(corrs[idx])
        if abs(corr) > best["best_channel_corr_abs"]:
            best.update({
                "best_channel_lag_samples": int(lag),
                "best_channel_corr_abs": float(abs(corr)),
                "best_channel_corr_signed": corr,
                "best_channel_idx": idx,
            })

        mean_view, peak_view = _shifted_views(rcg_abs_mean, rpeak_train, lag)
        corr_m = float(_corr_rows(mean_view, peak_view)[0])
        if abs(corr_m) > best["mean_abs_corr_abs"]:
            best.update({
                "mean_abs_lag_samples": int(lag),
                "mean_abs_corr_abs": float(abs(corr_m)),
                "mean_abs_corr_signed": corr_m,
            })

    return best


def _split_paths(cfg: MMECGConfig, protocol: str, fold_idx: int, split: str) -> list[Path]:
    if protocol == "samplewise":
        return [Path(cfg.samplewise_h5_dir) / f"{split}.h5"]
    if fold_idx == -1:
        return [Path(cfg.loso_h5_dir) / f"fold_{i:02d}" / f"{split}.h5" for i in range(1, cfg.n_folds + 1)]
    return [Path(cfg.loso_h5_dir) / f"fold_{fold_idx:02d}" / f"{split}.h5"]


def _collect_rows(paths: list[Path], max_lag_samples: int) -> pd.DataFrame:
    rows = []
    seg_id = 0
    for path in paths:
        with h5py.File(path, "r") as hf:
            n = hf["rcg"].shape[0]
            for i in range(n):
                rtrain = _gaussian_mask(hf["rpeak_indices"][i])
                lag_info = _best_input_lag(hf["rcg"][i], rtrain, max_lag_samples)
                row = {
                    "segment_id": seg_id,
                    "source_h5": str(path),
                    "record_id": hf["record_id"][i].decode() if isinstance(hf["record_id"][i], bytes) else str(hf["record_id"][i]),
                    "start_idx": int(hf["start_idx"][i]),
                    "end_idx": int(hf["end_idx"][i]),
                    "subject_id": int(hf["subject_id"][i]),
                    "scene": _decode_state(hf["physistatus"][i]),
                    **lag_info,
                }
                for prefix in ("best_channel", "mean_abs"):
                    lag = row[f"{prefix}_lag_samples"]
                    row[f"{prefix}_lag_ms"] = float(lag / FS * 1000.0)
                    row[f"{prefix}_abs_lag_ms"] = float(abs(lag) / FS * 1000.0)
                rows.append(row)
                seg_id += 1
    return pd.DataFrame(rows)


def _summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        df.groupby(keys)
        .agg(
            n=("segment_id", "size"),
            best_channel_lag_ms_median=("best_channel_lag_ms", "median"),
            best_channel_abs_lag_ms_mean=("best_channel_abs_lag_ms", "mean"),
            best_channel_corr_abs_mean=("best_channel_corr_abs", "mean"),
            best_channel_large_lag_rate=("best_channel_abs_lag_ms", lambda x: float((x >= 50).mean())),
            mean_abs_lag_ms_median=("mean_abs_lag_ms", "median"),
            mean_abs_abs_lag_ms_mean=("mean_abs_abs_lag_ms", "mean"),
            mean_abs_corr_abs_mean=("mean_abs_corr_abs", "mean"),
            mean_abs_large_lag_rate=("mean_abs_abs_lag_ms", lambda x: float((x >= 50).mean())),
        )
        .reset_index()
    )


def _json_default(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    return str(obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["samplewise", "loso"], default="samplewise")
    parser.add_argument("--fold_idx", type=int, default=1)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max_lag_ms", type=float, default=300.0)
    parser.add_argument("--out_tag", type=str, default=None)
    args = parser.parse_args()

    cfg = MMECGConfig()
    max_lag_samples = int(round(args.max_lag_ms / 1000.0 * FS))
    paths = _split_paths(cfg, args.protocol, args.fold_idx, args.split)
    tag = args.out_tag or (
        f"input_lag_{args.protocol}_{args.split}"
        if args.protocol == "samplewise"
        else f"input_lag_loso_fold{args.fold_idx:02d}_{args.split}"
    )
    out_dir = ROOT / "experiments_mmecg" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _collect_rows(paths, max_lag_samples)
    by_subject_scene = _summarize(df, ["subject_id", "scene"])
    by_subject = _summarize(df, ["subject_id"])
    by_scene = _summarize(df, ["scene"])

    df.to_csv(out_dir / "input_lag_metrics.csv", index=False)
    by_subject_scene.to_csv(out_dir / "input_lag_summary_by_subject_scene.csv", index=False)
    by_subject.to_csv(out_dir / "input_lag_summary_by_subject.csv", index=False)
    by_scene.to_csv(out_dir / "input_lag_summary_by_scene.csv", index=False)

    summary = {
        "protocol": args.protocol,
        "split": args.split,
        "fold_idx": args.fold_idx,
        "max_lag_ms": args.max_lag_ms,
        "n_segments": int(len(df)),
        "best_channel_lag_ms_median": float(df["best_channel_lag_ms"].median()),
        "best_channel_abs_lag_ms_mean": float(df["best_channel_abs_lag_ms"].mean()),
        "best_channel_corr_abs_mean": float(df["best_channel_corr_abs"].mean()),
        "best_channel_large_lag_rate_abs_ge_50ms": float((df["best_channel_abs_lag_ms"] >= 50).mean()),
        "mean_abs_lag_ms_median": float(df["mean_abs_lag_ms"].median()),
        "mean_abs_abs_lag_ms_mean": float(df["mean_abs_abs_lag_ms"].mean()),
        "mean_abs_corr_abs_mean": float(df["mean_abs_corr_abs"].mean()),
        "mean_abs_large_lag_rate_abs_ge_50ms": float((df["mean_abs_abs_lag_ms"] >= 50).mean()),
    }
    with open(out_dir / "input_lag_global_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    print(json.dumps(summary, indent=2, default=_json_default))
    print("\nBy subject-scene:")
    print(by_subject_scene.round(4).to_string(index=False))
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
