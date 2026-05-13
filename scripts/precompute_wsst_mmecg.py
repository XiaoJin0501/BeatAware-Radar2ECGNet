"""Pre-compute WSST spectrograms for all MMECG H5 splits.

Replicates the MATLAB MMECG_to_SST.m pipeline in Python:
  1. Resample each RCG bin from 200 Hz → 30 Hz (240 points for 8 s)
  2. Run synchrosqueezed CWT (Morlet, nv=10) → (72, 240)
  3. Take [:71, ::2] → (71, 120), normalise to [0, 1]

Output: for each input `<dir>/{split}.h5`, writes `<dir>/{split}_wsst.h5`
with a single dataset 'wsst' of shape (N, 50, 71, 120) float32.

Usage:
  # All samplewise splits
  python scripts/precompute_wsst_mmecg.py --mode samplewise

  # All loso folds (takes ~2h on a single core, use --workers N to parallelise)
  python scripts/precompute_wsst_mmecg.py --mode loso --workers 4

  # Single file
  python scripts/precompute_wsst_mmecg.py \
      --files /path/to/train.h5 /path/to/val.h5
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
from scipy.signal import resample
from ssqueezepy import ssq_cwt


# ─── WSST transform (per range bin) ──────────────────────────────────────────

def _bin_to_wsst(x_200hz: np.ndarray) -> np.ndarray:
    """(1600,) float32 at 200 Hz → (71, 120) WSST magnitude [0, 1]."""
    x_30 = resample(x_200hz.astype(np.float64), 240)
    Tx, _, _, _ = ssq_cwt(x_30, wavelet="morlet", fs=30, nv=10)
    mag = np.abs(Tx[:71, ::2]).astype(np.float32)   # (71, 120)
    vmax = float(mag.max())
    if vmax > 1e-8:
        mag /= vmax
    return mag


def window_to_wsst(rcg_window: np.ndarray) -> np.ndarray:
    """(50, 1600) float32 → (50, 71, 120) WSST magnitude [0, 1]."""
    return np.stack([_bin_to_wsst(rcg_window[b]) for b in range(rcg_window.shape[0])])


# ─── Per-file worker ──────────────────────────────────────────────────────────

def precompute_h5(h5_path: str | Path, overwrite: bool = False) -> str:
    """Compute WSST for all windows in *h5_path* and write companion *_wsst.h5*."""
    h5_path = Path(h5_path)
    out_path = h5_path.with_name(h5_path.stem + "_wsst.h5")

    if out_path.exists() and not overwrite:
        return f"SKIP  {out_path.name} (already exists)"

    t0 = time.time()
    with h5py.File(h5_path, "r") as fin:
        rcg = fin["rcg"][:]          # (N, 50, 1600)
    N = rcg.shape[0]

    wsst_arr = np.empty((N, 50, 71, 120), dtype=np.float16)  # float16 halves storage
    for i in range(N):
        wsst_arr[i] = window_to_wsst(rcg[i]).astype(np.float16)
        if (i + 1) % 50 == 0 or i == N - 1:
            pct = (i + 1) / N * 100
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N - i - 1)
            print(f"  {h5_path.name}: {i+1}/{N} ({pct:.0f}%)  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    with h5py.File(out_path, "w") as fout:
        fout.create_dataset("wsst", data=wsst_arr,
                            chunks=(1, 50, 71, 120), compression="lzf")

    elapsed = time.time() - t0
    sz_mb = out_path.stat().st_size / 1e6
    return f"DONE  {out_path.name}  N={N}  {elapsed:.0f}s  {sz_mb:.0f} MB"


# ─── CLI ─────────────────────────────────────────────────────────────────────

def collect_h5_paths(mode: str) -> list[Path]:
    base = Path("/home/qhh2237/Datasets/MMECG/processed")
    paths: list[Path] = []
    if mode == "samplewise":
        for split in ("train", "val", "test"):
            p = base / "samplewise" / f"{split}.h5"
            if p.exists():
                paths.append(p)
    elif mode == "loso":
        for fold in range(1, 12):
            fold_dir = base / f"loso/fold_{fold:02d}"
            for split in ("train", "val", "test"):
                p = fold_dir / f"{split}.h5"
                if p.exists():
                    paths.append(p)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["samplewise", "loso", "files"],
                        default="samplewise")
    parser.add_argument("--files", nargs="+", type=str,
                        help="Explicit H5 file paths (used when --mode files)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (each processes one file)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if _wsst.h5 already exists")
    args = parser.parse_args()

    if args.mode == "files":
        if not args.files:
            parser.error("--files required when --mode files")
        paths = [Path(p) for p in args.files]
    else:
        paths = collect_h5_paths(args.mode)

    print(f"Files to process: {len(paths)}")
    for p in paths:
        print(f"  {p}")

    if args.workers <= 1 or len(paths) == 1:
        for p in paths:
            print(f"\nProcessing {p.name} ...")
            msg = precompute_h5(p, overwrite=args.overwrite)
            print(msg)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(precompute_h5, p, args.overwrite): p for p in paths}
            for fut in as_completed(futs):
                print(fut.result())

    print("\nAll done.")


if __name__ == "__main__":
    main()
