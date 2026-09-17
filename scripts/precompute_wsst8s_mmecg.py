#!/usr/bin/env python3
"""Build a deduplicated subject-centric full-8-second MMECG WSST pool.

The transform is intentionally identical to commit 1aaf8b7's original
`precompute_wsst8s_mmecg.py`: Morlet SSQ-CWT, magnitude, 71 frequency bins,
240 time bins, per-map [0,1] normalization, and float16 storage. Existing
fold caches are copied byte-for-byte when available; only missing unique
windows are transformed. No 4-second input is accepted.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import resample
from ssqueezepy import ssq_cwt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.wsst8s_subject_pool import SUPPORT, window_key  # noqa: E402


DATA_ROOT = Path("/home/qhh2237/Datasets/MMECG/processed")
SPLIT_ROOT = DATA_ROOT / "loso_v2"
POOL_DIR = DATA_ROOT / "wsst8s_subject_pool"
WIN_SAMPLES = 1600
N_TIME_BINS = 240
N_FREQ_BINS = 71
HOP_SAMPLES = WIN_SAMPLES / N_TIME_BINS
RAW_BYTES_PER_WINDOW = 50 * N_FREQ_BINS * N_TIME_BINS * np.dtype(np.float16).itemsize
MIN_FREE_AFTER = 8 * 2**30

assert abs(HOP_SAMPLES - 800 / 120) < 1e-9


def normalize_01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def wsst_full_one_bin(signal: np.ndarray, fs: int = 200, nv: int = 16) -> np.ndarray:
    if signal.shape[-1] != WIN_SAMPLES:
        raise ValueError(f"Expected {WIN_SAMPLES} samples, got {signal.shape[-1]}")
    tx, *_ = ssq_cwt(signal.astype(np.float64), wavelet="morlet", fs=fs, nv=nv)
    mag = np.abs(tx).astype(np.float32)
    start = mag.shape[0] // 2 if mag.shape[0] >= 2 * N_FREQ_BINS else 0
    mag = mag[start:start + N_FREQ_BINS]
    if mag.shape[0] < N_FREQ_BINS:
        mag = np.concatenate((mag, np.zeros(
            (N_FREQ_BINS - mag.shape[0], mag.shape[1]), dtype=np.float32)))
    mag = resample(mag, N_TIME_BINS, axis=1).astype(np.float32)
    return normalize_01(np.clip(mag, 0.0, None))


def transform_window(payload: tuple[np.ndarray, int, int]) -> np.ndarray:
    """Pure per-window transform; safe to run in independent worker processes."""
    rcg, fs, nv = payload
    result = np.empty((50, N_FREQ_BINS, N_TIME_BINS), dtype=np.float16)
    for channel in range(50):
        result[channel] = wsst_full_one_bin(rcg[channel], fs, nv).astype(np.float16)
    return result


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def inventory() -> dict[int, dict[tuple[int, str, int, int], tuple[Path, int]]]:
    """Map every unique window to one authoritative base-H5 source row."""
    result: dict[int, dict[tuple[int, str, int, int], tuple[Path, int]]] = defaultdict(dict)
    for fold in range(1, 12):
        for split in ("train", "val", "test"):
            path = SPLIT_ROOT / f"fold_{fold:02d}" / f"{split}.h5"
            with h5py.File(path, "r") as h:
                if h["rcg"].shape[1:] != (50, WIN_SAMPLES):
                    raise ValueError(f"Unexpected RCG shape in {path}: {h['rcg'].shape}")
                for row, values in enumerate(zip(
                        h["subject_id"][:], h["record_id"][:],
                        h["start_idx"][:], h["end_idx"][:])):
                    key = window_key(*values)
                    result[key[0]].setdefault(key, (path, row))
    return result


def cached_lookup() -> dict[tuple[int, str, int, int], tuple[Path, int]]:
    """Index existing verified 8-second fold caches for byte-identical reuse."""
    result = {}
    for wsst_path in sorted(SPLIT_ROOT.glob("fold_*/*_wsst8s.h5")):
        split = wsst_path.name.removesuffix("_wsst8s.h5")
        base_path = wsst_path.with_name(f"{split}.h5")
        with h5py.File(wsst_path, "r") as w, h5py.File(base_path, "r") as h:
            if w.attrs.get("support") != SUPPORT or w["wsst"].shape[0] != h["rcg"].shape[0]:
                raise ValueError(f"Invalid reusable cache {wsst_path}")
            for row, values in enumerate(zip(
                    h["subject_id"][:], h["record_id"][:],
                    h["start_idx"][:], h["end_idx"][:])):
                result.setdefault(window_key(*values), (wsst_path, row))
    return result


def ensure_space(unique_windows: int) -> None:
    usage = shutil.disk_usage(POOL_DIR.parent)
    # Raw float16 is a conservative ceiling; HDF5 LZF should be smaller.
    ceiling = unique_windows * (RAW_BYTES_PER_WINDOW + WIN_SAMPLES * 4)
    if usage.free - ceiling < MIN_FREE_AFTER:
        raise SystemExit(
            f"Insufficient disk: {usage.free/2**30:.1f} GiB free, conservative pool "
            f"ceiling {ceiling/2**30:.1f} GiB, require {MIN_FREE_AFTER/2**30:.0f} GiB reserve")


def write_subject(subject: int, sources, reusable, fs: int, nv: int, workers: int) -> None:
    output = POOL_DIR / f"subject_{subject:02d}.h5"
    if output.exists():
        print(f"[skip] {output}", flush=True)
        return
    keys = sorted(sources)
    tmp = output.with_suffix(".tmp.h5")
    if tmp.exists():
        tmp.unlink()  # incomplete temporary file only; never a completed artifact
    str_dtype = h5py.string_dtype("utf-8")
    reused = computed = 0
    started = time.time()
    with h5py.File(tmp, "w") as dst:
        shape = (len(keys), 50, N_FREQ_BINS, N_TIME_BINS)
        out = dst.create_dataset("wsst", shape=shape, dtype=np.float16,
                                 compression="lzf", chunks=(1, 50, N_FREQ_BINS, N_TIME_BINS))
        ecg_out = dst.create_dataset("ecg", shape=(len(keys), 1, WIN_SAMPLES),
                                     dtype=np.float32, compression="lzf")
        dst.create_dataset("subject_id", data=np.asarray([k[0] for k in keys], np.int32))
        dst.create_dataset("record_id", data=np.asarray([k[1] for k in keys], dtype=object), dtype=str_dtype)
        dst.create_dataset("start_idx", data=np.asarray([k[2] for k in keys], np.int32))
        dst.create_dataset("end_idx", data=np.asarray([k[3] for k in keys], np.int32))
        open_base = open_cache = None
        base_path = cache_path = None
        try:
            pending: list[tuple[int, np.ndarray]] = []

            def flush_pending(executor: ProcessPoolExecutor) -> None:
                nonlocal computed
                if not pending:
                    return
                payloads = [(rcg, fs, nv) for _, rcg in pending]
                for (output_row, _), value in zip(pending, executor.map(transform_window, payloads)):
                    out[output_row] = value
                    computed += 1
                pending.clear()

            with ProcessPoolExecutor(max_workers=workers) as executor:
                for i, key in enumerate(keys):
                    src_path, src_row = sources[key]
                    if src_path != base_path:
                        if open_base is not None: open_base.close()
                        open_base = h5py.File(src_path, "r"); base_path = src_path
                    ecg_out[i] = open_base["ecg"][src_row]
                    if key in reusable:
                        wp, wr = reusable[key]
                        if wp != cache_path:
                            if open_cache is not None: open_cache.close()
                            open_cache = h5py.File(wp, "r"); cache_path = wp
                        out[i] = open_cache["wsst"][wr]
                        reused += 1
                    else:
                        pending.append((i, np.asarray(open_base["rcg"][src_row], dtype=np.float32)))
                        if len(pending) >= workers * 2:
                            flush_pending(executor)
                    if (i + 1) % 25 == 0 or i + 1 == len(keys):
                        flush_pending(executor)
                        print(f"subject {subject:02d}: {i+1}/{len(keys)} reused={reused} computed={computed}", flush=True)
        finally:
            if open_base is not None: open_base.close()
            if open_cache is not None: open_cache.close()
        attrs = {
            "subject_id": subject, "support": SUPPORT, "crop": "none",
            "supported_fraction_of_target": 1.0, "window_samples": WIN_SAMPLES,
            "n_time_bins": N_TIME_BINS, "n_freq_bins": N_FREQ_BINS,
            "hop_samples": HOP_SAMPLES, "hop_ms": 1000 * HOP_SAMPLES / fs,
            "fs": fs, "nv": nv, "wavelet": "morlet",
            "normalization": "per-channel per-window minmax after magnitude CWT",
            "dtype": "float16", "generator": "scripts/precompute_wsst8s_mmecg.py",
            "reference_generator_commit": "1aaf8b79950c30c770d1560e893879b68890d602",
            "code_commit": git_commit(), "created_utc": datetime.now(timezone.utc).isoformat(),
            "unique_windows": len(keys), "reused_windows": reused, "computed_windows": computed,
        }
        for key, value in attrs.items(): dst.attrs[key] = value
    os.replace(tmp, output)
    print(f"[done] {output} {output.stat().st_size/2**30:.2f} GiB in {(time.time()-started)/60:.1f} min", flush=True)


def validate_bitwise(pool_dir: Path, reusable) -> None:
    checked = 0
    for path in sorted(pool_dir.glob("subject_*.h5")):
        with h5py.File(path, "r") as h:
            for row, values in enumerate(zip(h["subject_id"][:], h["record_id"][:],
                                             h["start_idx"][:], h["end_idx"][:])):
                key = window_key(*values)
                if key not in reusable: continue
                old_path, old_row = reusable[key]
                with h5py.File(old_path, "r") as old:
                    if not np.array_equal(h["wsst"][row], old["wsst"][old_row]):
                        raise AssertionError(f"Bitwise mismatch for {key}")
                checked += 1
                if checked >= 64:
                    print(f"PASS: {checked} reused windows bit-identical", flush=True)
                    return
    raise RuntimeError("No reusable windows available for bitwise validation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default="all", help="all or comma-separated subject IDs")
    parser.add_argument("--fs", type=int, default=200)
    parser.add_argument("--nv", type=int, default=16)
    parser.add_argument("--inventory_only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    sources = inventory()
    selected = sorted(sources) if args.subjects == "all" else [int(x) for x in args.subjects.split(",")]
    unique = sum(len(sources[s]) for s in selected)
    ensure_space(unique)
    reusable = cached_lookup()
    print(f"subjects={selected} unique_windows={unique} reusable={sum(k in reusable for s in selected for k in sources[s])}", flush=True)
    if args.inventory_only: return
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    for subject in selected:
        write_subject(subject, sources[subject], reusable, args.fs, args.nv, args.workers)
    validate_bitwise(POOL_DIR, reusable)
    size = sum(p.stat().st_size for p in POOL_DIR.glob("subject_*.h5"))
    print(f"POOL COMPLETE: {size} bytes = {size/2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
