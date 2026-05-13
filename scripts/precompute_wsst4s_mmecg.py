#!/usr/bin/env python3
"""Precompute paper-style 4-second WSST tensors for MMECG radarODE-MTL runs.

The original radarODE-MTL preprocessing uses 4-second radar windows and stores
time-frequency maps with shape (range_bin=50, freq=71, time=120).  This script
creates companion H5 files next to the existing split files:

    train.h5 -> train_wsst4s.h5
    val.h5   -> val_wsst4s.h5
    test.h5  -> test_wsst4s.h5

It keeps the original split files untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy.signal import resample
from tqdm import tqdm

try:
    from ssqueezepy import ssq_cwt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ssqueezepy is required: pip install ssqueezepy") from exc


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/qhh2237/Datasets/MMECG/processed")


def _split_files(mode: str, fold_idx: int | None = None) -> list[Path]:
    if mode == "loso":
        files: list[Path] = []
        folds = [fold_idx] if fold_idx is not None else list(range(1, 12))
        for fold in folds:
            if fold is None or fold < 1 or fold > 11:
                raise ValueError(f"fold_idx must be 1..11, got {fold}")
            fold_dir = DATA_ROOT / "loso" / f"fold_{fold:02d}"
            files.extend([fold_dir / "train.h5", fold_dir / "val.h5", fold_dir / "test.h5"])
        return files
    if mode == "samplewise":
        split_dir = DATA_ROOT / "samplewise"
        return [split_dir / "train.h5", split_dir / "val.h5", split_dir / "test.h5"]
    raise ValueError(f"Unsupported mode: {mode}")


def _normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max - x_min < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def _wsst_4s_one_bin(signal_4s: np.ndarray, fs: int, nv: int) -> np.ndarray:
    """Return one bin's WSST map with shape (71, 120)."""
    tx, *_ = ssq_cwt(signal_4s.astype(np.float64), wavelet="morlet", fs=fs, nv=nv)
    mag = np.abs(tx).astype(np.float32)

    # The MATLAB pipeline keeps the upper half of the frequency axis and uses 71
    # frequency bins.  ssqueezepy can return slightly different frequency counts
    # across versions, so this preserves the same intent with padding fallback.
    start = mag.shape[0] // 2 if mag.shape[0] >= 142 else 0
    mag = mag[start : start + 71]
    if mag.shape[0] < 71:
        pad = np.zeros((71 - mag.shape[0], mag.shape[1]), dtype=np.float32)
        mag = np.concatenate([mag, pad], axis=0)

    mag = resample(mag, 120, axis=1).astype(np.float32)
    mag = np.clip(mag, 0.0, None)
    return _normalize_01(mag)


def _precompute_file(path: Path, overwrite: bool, batch_size: int, fs: int, nv: int) -> None:
    out_path = path.with_name(f"{path.stem}_wsst4s.h5")
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path}")
        return
    if not path.exists():
        print(f"[missing] {path}", file=sys.stderr)
        return

    with h5py.File(path, "r") as src:
        if "rcg" not in src:
            raise KeyError(f"{path} does not contain dataset 'rcg'")
        n = int(src["rcg"].shape[0])
        if src["rcg"].shape[-1] < 1200:
            raise ValueError(f"{path} rcg length is too short for center 4s crop")

        tmp_path = out_path.with_suffix(".tmp.h5")
        with h5py.File(tmp_path, "w") as dst:
            dst.create_dataset("wsst", shape=(n, 50, 71, 120), dtype=np.float16, compression="lzf")
            dst.attrs["source_h5"] = str(path)
            dst.attrs["crop"] = "center_4s_samples_400_1200"
            dst.attrs["fs"] = fs
            dst.attrs["nv"] = nv
            dst.attrs["shape"] = "(N,50,71,120)"

            pbar = tqdm(range(0, n, batch_size), desc=out_path.name, leave=False)
            for start in pbar:
                stop = min(start + batch_size, n)
                rcg = np.asarray(src["rcg"][start:stop], dtype=np.float32)
                rcg_4s = rcg[..., 400:1200]
                out = np.empty((stop - start, 50, 71, 120), dtype=np.float16)
                for i in range(stop - start):
                    for b in range(50):
                        out[i, b] = _wsst_4s_one_bin(rcg_4s[i, b], fs=fs, nv=nv).astype(np.float16)
                dst["wsst"][start:stop] = out

        tmp_path.replace(out_path)
        print(f"[done] {out_path} ({n} samples)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["loso", "samplewise"], default="loso")
    parser.add_argument("--fold_idx", type=int, default=None, help="Optional LOSO fold to process first.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--fs", type=int, default=200)
    parser.add_argument("--nv", type=int, default=10)
    args = parser.parse_args()

    for path in _split_files(args.mode, args.fold_idx):
        _precompute_file(path, overwrite=args.overwrite, batch_size=args.batch_size, fs=args.fs, nv=args.nv)


if __name__ == "__main__":
    main()
