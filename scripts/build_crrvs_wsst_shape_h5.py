#!/usr/bin/env python3
"""Build MMECG-compatible CW-STFT H5 splits for CR_RVS external validation.

The output layout mirrors MMECG LOSO enough to reuse `train_mmecg.py` with
REM-AlignNet-MTL:

    processed/q1_shape_kfold/fold_01/{train,val,test}.h5
    processed/q1_shape_kfold/fold_01/{train,val,test}_wsst.h5

Radar input is 24-GHz CW I/Q converted to an unwrapped phase/displacement-like
signal, downsampled to 200 Hz, then represented as log-STFT `[1,F,T]`.
ECG target is Lead II (`tfm_ecg2`) downsampled to 200 Hz and windowed into
8-second segments. R-peaks are detected from ECG and used only for labels and
the shape-200 target crop, as in the MMECG shape-head evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
CR_ROOT = Path("/home/qhh2237/Datasets/CR_RVS")
STATE_MAP = {"Resting": b"NB", "Valsalva": b"IB", "Apnea": b"SP"}


def _load_mat(path: Path) -> dict[str, Any]:
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    return {
        "radar_i": np.asarray(mat["radar_i"], dtype=np.float64).reshape(-1),
        "radar_q": np.asarray(mat["radar_q"], dtype=np.float64).reshape(-1),
        "ecg": np.asarray(mat["tfm_ecg2"], dtype=np.float64).reshape(-1),
        "fs_radar": float(np.asarray(mat["fs_radar"]).reshape(-1)[0]),
        "fs_ecg": float(np.asarray(mat["fs_ecg"]).reshape(-1)[0]),
    }


def _subject_int(subject_id: str) -> int:
    m = re.search(r"(\d+)$", subject_id)
    return int(m.group(1)) if m else 0


def _downsample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-6:
        return x.astype(np.float32)
    up = int(round(fs_out))
    down = int(round(fs_in))
    g = np.gcd(up, down)
    y = signal.resample_poly(x, up // g, down // g)
    return y.astype(np.float32)


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    hi = min(hi, fs / 2.0 - 1e-3)
    b, a = signal.butter(order, [lo, hi], btype="bandpass", fs=fs)
    return signal.filtfilt(b, a, x).astype(np.float32)


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    y = (x - med) / (1.4826 * mad)
    return np.clip(y, -8.0, 8.0).astype(np.float32)


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _stft_map(x: np.ndarray, fs: float, nperseg: int, noverlap: int) -> np.ndarray:
    _, _, z = signal.stft(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    mag = np.log1p(np.abs(z)).astype(np.float32)
    mag = (mag - mag.mean()) / (mag.std() + 1e-6)
    return mag[np.newaxis, :, :].astype(np.float16)


def _detect_rpeaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    try:
        cleaned = nk.ecg_clean(ecg, sampling_rate=int(round(fs)), method="neurokit")
        _, info = nk.ecg_peaks(cleaned, sampling_rate=int(round(fs)), method="neurokit")
        r = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64)
    except Exception:
        distance = int(round(0.35 * fs))
        height = np.percentile(ecg, 75)
        r, _ = signal.find_peaks(ecg, distance=distance, height=height)
        r = r.astype(np.int64)
    return np.sort(np.unique(r[(r >= 0) & (r < len(ecg))]))


def _center_cycle_bounds(rpeaks: np.ndarray, win_start: int, win_len: int) -> tuple[int, int] | None:
    local = np.asarray(rpeaks[(rpeaks >= win_start) & (rpeaks < win_start + win_len)] - win_start, dtype=np.int64)
    if local.size < 2:
        return None
    center = win_len // 2
    pairs = [(int(a), int(b)) for a, b in zip(local[:-1], local[1:]) if int(b) > int(a) + 10]
    if not pairs:
        return None
    containing = [p for p in pairs if p[0] <= center <= p[1]]
    if containing:
        return containing[0]
    return min(pairs, key=lambda p: abs(((p[0] + p[1]) / 2.0) - center))


def _make_windows(record: pd.Series, args: argparse.Namespace) -> list[dict[str, Any]]:
    data = _load_mat(Path(record["file_path"]))
    fs = float(args.fs)

    iq = data["radar_i"] + 1j * data["radar_q"]
    phase = np.unwrap(np.angle(iq))
    phase = signal.detrend(phase).astype(np.float32)

    phase = _downsample(phase, data["fs_radar"], fs)
    ecg = _downsample(data["ecg"], data["fs_ecg"], fs)
    n = min(len(phase), len(ecg))
    phase, ecg = phase[:n], ecg[:n]
    phase = _bandpass(phase, fs, args.radar_lo_hz, args.radar_hi_hz)
    ecg = _bandpass(ecg, fs, args.ecg_lo_hz, args.ecg_hi_hz)
    rpeaks = _detect_rpeaks(ecg, fs)
    if args.ecg_scale == "zscore_record":
        # Continuous contract: the target keeps within-record amplitude and baseline, so it
        # is z-scored once over the whole record (the MMECG convention) instead of min-max
        # normalised per window, which discards both (CONTINUOUS_ECG_TASK_CONTRACT, sec. 3).
        ecg = ((ecg - ecg.mean()) / (ecg.std() + 1e-8)).astype(np.float32)

    win_len = int(round(args.window_sec * fs))
    stride = int(round(args.stride_sec * fs))
    if n < win_len or rpeaks.size < 3:
        return []
    starts = np.arange(0, n - win_len + 1, stride, dtype=np.int64)
    if args.max_windows_per_record and len(starts) > args.max_windows_per_record:
        rng = np.random.default_rng(args.seed + _subject_int(str(record["subject_id"])) + int(record["scenario_id"]))
        starts = np.sort(rng.choice(starts, size=args.max_windows_per_record, replace=False))

    rows: list[dict[str, Any]] = []
    for start in starts:
        bounds = _center_cycle_bounds(rpeaks, int(start), win_len)
        if bounds is None:
            continue
        radar_win = _robust_z(phase[start : start + win_len])
        ecg_win = (ecg[start : start + win_len] if args.ecg_scale == "zscore_record"
                   else _minmax(ecg[start : start + win_len]))[np.newaxis, :].astype(np.float32)
        local_r = (rpeaks[(rpeaks >= start) & (rpeaks < start + win_len)] - start).astype(np.int32)
        rows.append(
            {
                "radar": radar_win[np.newaxis, :].astype(np.float32),
                "ecg": ecg_win.astype(np.float32),
                "rpeak_indices": local_r,
                "cycle_bounds": np.asarray(bounds, dtype=np.int32),
                "wsst": _stft_map(radar_win, fs, args.stft_nperseg, args.stft_noverlap),
                "subject_id": _subject_int(str(record["subject_id"])),
                "physistatus": STATE_MAP.get(str(record["scenario_name"]), b"NB"),
                "record_id": str(record["record_id"]).encode(),
                "file_name": str(record["file_name"]).encode(),
                "start_idx": int(start),
                "end_idx": int(start + win_len),
            }
        )
    return rows


def _write_split(rows: list[dict[str, Any]], h5_path: Path, wsst_path: Path,
                 wsst_attrs: dict[str, Any] | None = None) -> None:
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    vlen_i32 = h5py.vlen_dtype(np.dtype("int32"))
    str_dt = h5py.string_dtype(encoding="utf-8")
    n = len(rows)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("rcg", data=np.stack([r["radar"] for r in rows], axis=0), compression="gzip")
        f.create_dataset("ecg", data=np.stack([r["ecg"] for r in rows], axis=0), compression="gzip")
        f.create_dataset("subject_id", data=np.asarray([r["subject_id"] for r in rows], dtype=np.int32))
        f.create_dataset("physistatus", data=np.asarray([r["physistatus"] for r in rows], dtype=object), dtype=str_dt)
        f.create_dataset("start_idx", data=np.asarray([r["start_idx"] for r in rows], dtype=np.int32))
        f.create_dataset("end_idx", data=np.asarray([r["end_idx"] for r in rows], dtype=np.int32))
        f.create_dataset("record_id", data=np.asarray([r["record_id"] for r in rows], dtype=object), dtype=str_dt)
        f.create_dataset("file_name", data=np.asarray([r["file_name"] for r in rows], dtype=object), dtype=str_dt)
        f.create_dataset("delineation_valid", data=np.ones(n, dtype=np.uint8))
        f.create_dataset("cycle_bounds", data=np.stack([r["cycle_bounds"] for r in rows], axis=0).astype(np.int32))
        for key in ("rpeak_indices", "q_indices", "s_indices", "tpeak_indices"):
            ds = f.create_dataset(key, shape=(n,), dtype=vlen_i32)
            for i, row in enumerate(rows):
                ds[i] = row["rpeak_indices"] if key == "rpeak_indices" else np.asarray([], dtype=np.int32)
    with h5py.File(wsst_path, "w") as f:
        f.create_dataset("wsst", data=np.stack([r["wsst"] for r in rows], axis=0), compression="gzip")
        for key, value in (wsst_attrs or {}).items():
            f.attrs[key] = value


def build(args: argparse.Namespace) -> dict[str, Any]:
    splits = pd.read_csv(args.splits_csv)
    splits = splits[splits["scenario_name"].isin(["Resting", "Valsalva", "Apnea"])].copy()
    if args.folds:
        splits = splits[splits["fold_id"].isin(args.folds)]
    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = CR_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {"output_dir": str(out_root), "folds": {}, "args": vars(args)}
    for fold in sorted(splits["fold_id"].unique()):
        fold_rows = splits[splits["fold_id"] == fold]
        manifest["folds"][str(int(fold))] = {}
        for split in ("train", "val", "test"):
            recs = fold_rows[fold_rows["split"] == split]
            if args.limit_records_per_split:
                recs = recs.head(args.limit_records_per_split)
            rows: list[dict[str, Any]] = []
            for _, rec in tqdm(recs.iterrows(), total=len(recs), desc=f"fold {fold} {split}"):
                try:
                    rows.extend(_make_windows(rec, args))
                except Exception as exc:
                    print(f"[WARN] skipped {rec.get('file_name')}: {exc}", file=sys.stderr)
            if not rows:
                raise RuntimeError(f"No windows generated for fold {fold} split {split}")
            fold_dir = out_root / f"fold_{int(fold):02d}"
            if args.ecg_scale == "zscore_record":
                # The continuous loader asserts full-window support; the STFT frames tile
                # samples 0..1599 without padding, so the whole 8 s target is covered.
                wsst_file = fold_dir / f"{split}_wsst8s.h5"
                wsst_attrs = {"support": "full_window_samples_0_1600",
                              "supported_fraction_of_target": 1.0,
                              "representation": "stft_log_magnitude",
                              "stft_nperseg": args.stft_nperseg,
                              "stft_noverlap": args.stft_noverlap}
            else:
                wsst_file, wsst_attrs = fold_dir / f"{split}_wsst.h5", None
            _write_split(rows, fold_dir / f"{split}.h5", wsst_file, wsst_attrs)
            manifest["folds"][str(int(fold))][split] = {
                "records": int(len(recs)),
                "windows": int(len(rows)),
                "h5": str(fold_dir / f"{split}.h5"),
                "wsst": str(wsst_file),
            }
            print(f"fold {fold} {split}: records={len(recs)} windows={len(rows)}")
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits_csv", default=str(CR_ROOT / "metadata" / "crrvs_kfold_splits.csv"))
    parser.add_argument("--output_dir", default=str(CR_ROOT / "processed" / "q1_shape_kfold"))
    parser.add_argument("--folds", type=int, nargs="*", default=[1])
    parser.add_argument("--limit_records_per_split", type=int, default=0)
    parser.add_argument("--max_windows_per_record", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fs", type=float, default=200.0)
    parser.add_argument("--window_sec", type=float, default=8.0)
    parser.add_argument("--stride_sec", type=float, default=4.0)
    parser.add_argument("--radar_lo_hz", type=float, default=0.6)
    parser.add_argument("--radar_hi_hz", type=float, default=12.0)
    parser.add_argument("--ecg_lo_hz", type=float, default=0.5)
    parser.add_argument("--ecg_hi_hz", type=float, default=40.0)
    parser.add_argument("--stft_nperseg", type=int, default=128)
    parser.add_argument("--stft_noverlap", type=int, default=112)
    parser.add_argument("--ecg_scale", choices=("minmax_window", "zscore_record"),
                        default="minmax_window",
                        help="minmax_window reproduces the Shape-200 set; zscore_record builds the "
                             "continuous-contract target")
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps(manifest["folds"], indent=2))


if __name__ == "__main__":
    main()
