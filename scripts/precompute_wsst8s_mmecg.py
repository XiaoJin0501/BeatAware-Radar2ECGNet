#!/usr/bin/env python3
"""Recompute MMECG WSST over the FULL 8 s / 1600-sample radar support.

Replaces the centre-4 s crop of `precompute_wsst4s_mmecg.py`, under which half of
every 1600-sample ECG target had no radar input support (docs/RCG_PROVENANCE_AUDIT.md).

The temporal hop is PRESERVED, not stretched: the 4 s pipeline mapped 800 samples to
120 bins, i.e. 6.667 samples (33.3 ms) per bin. Over 1600 samples that same hop gives
240 bins. We therefore emit (50, 71, 240) and never resample 120 bins across 8 s.

Writes `<split>_wsst8s.h5`. Existing `_wsst.h5` / `_wsst4s.h5` are never modified.
"""
from __future__ import annotations

import argparse, hashlib, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import resample
from tqdm import tqdm

from ssqueezepy import ssq_cwt

DATA_ROOT = Path("/home/qhh2237/Datasets/MMECG/processed")
WIN_SAMPLES = 1600            # full window, matches the ECG target exactly
N_TIME_BINS = 240             # 1600 / (800/120) -> identical hop to the 4 s pipeline
N_FREQ_BINS = 71
HOP_SAMPLES = WIN_SAMPLES / N_TIME_BINS   # 6.666..., as in the 4 s pipeline

# ---- hard support-alignment assertions (fail loudly, never silently degrade) ----
_LEGACY_SAMPLES, _LEGACY_BINS = 800, 120
assert abs(WIN_SAMPLES / N_TIME_BINS - _LEGACY_SAMPLES / _LEGACY_BINS) < 1e-9, \
    "temporal hop must match the legacy 4 s pipeline exactly"
assert N_TIME_BINS == round(WIN_SAMPLES * _LEGACY_BINS / _LEGACY_SAMPLES), \
    "bin count must follow from preserving the hop, not from stretching"


def _normalize_01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _wsst_full_one_bin(sig: np.ndarray, fs: int, nv: int) -> np.ndarray:
    """One range bin -> (71, 240) over the FULL window."""
    assert sig.shape[-1] == WIN_SAMPLES, f"expected {WIN_SAMPLES} samples, got {sig.shape[-1]}"
    tx, *_ = ssq_cwt(sig.astype(np.float64), wavelet="morlet", fs=fs, nv=nv)
    mag = np.abs(tx).astype(np.float32)
    start = mag.shape[0] // 2 if mag.shape[0] >= 2 * N_FREQ_BINS else 0
    mag = mag[start : start + N_FREQ_BINS]
    if mag.shape[0] < N_FREQ_BINS:
        mag = np.concatenate([mag, np.zeros((N_FREQ_BINS - mag.shape[0], mag.shape[1]), np.float32)])
    mag = resample(mag, N_TIME_BINS, axis=1).astype(np.float32)
    return _normalize_01(np.clip(mag, 0.0, None))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _precompute(path: Path, overwrite: bool, batch: int, fs: int, nv: int) -> None:
    out = path.with_name(f"{path.stem}_wsst8s.h5")
    if out.exists() and not overwrite:
        print(f"[skip] {out}"); return
    if not path.exists():
        print(f"[missing] {path}", file=sys.stderr); return

    with h5py.File(path, "r") as src:
        n = int(src["rcg"].shape[0])
        n_bins = int(src["rcg"].shape[1])
        # ---- support-alignment assertions against the actual data ----
        assert src["rcg"].shape[-1] == WIN_SAMPLES, \
            f"rcg length {src['rcg'].shape[-1]} != window {WIN_SAMPLES}"
        assert src["ecg"].shape[-1] == WIN_SAMPLES, \
            f"ecg length {src['ecg'].shape[-1]} != window {WIN_SAMPLES}; radar and target must span the same range"
        tmp = out.with_suffix(".tmp.h5")
        t0 = time.time()
        with h5py.File(tmp, "w") as dst:
            d = dst.create_dataset("wsst", shape=(n, n_bins, N_FREQ_BINS, N_TIME_BINS),
                                   dtype=np.float16, compression="lzf")
            prov = {
                "source_h5": str(path),
                "support": "full_window_samples_0_1600",
                "crop": "none",
                "supported_fraction_of_target": 1.0,
                "window_samples": WIN_SAMPLES,
                "n_time_bins": N_TIME_BINS,
                "n_freq_bins": N_FREQ_BINS,
                "hop_samples": HOP_SAMPLES,
                "hop_ms": 1000.0 * HOP_SAMPLES / fs,
                "hop_preserved_from": "precompute_wsst4s_mmecg.py (800 samples -> 120 bins)",
                "fs": fs, "nv": nv, "wavelet": "morlet",
                "shape": f"(N,{n_bins},{N_FREQ_BINS},{N_TIME_BINS})",
                "narrow_bandpass_applied": False,
                "topk_method": "none (all range bins retained)",
                "generator": "scripts/precompute_wsst8s_mmecg.py",
                "git_commit": _git_commit(),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "supersedes": "center_4s_samples_400_1200 (protocol-confounded: 50% of target unsupported)",
            }
            for k, v in prov.items():
                dst.attrs[k] = v
            for lo in tqdm(range(0, n, batch), desc=out.name, leave=False):
                hi = min(lo + batch, n)
                rcg = np.asarray(src["rcg"][lo:hi], dtype=np.float32)
                buf = np.empty((hi - lo, n_bins, N_FREQ_BINS, N_TIME_BINS), np.float16)
                for i in range(hi - lo):
                    for b in range(n_bins):
                        buf[i, b] = _wsst_full_one_bin(rcg[i, b], fs=fs, nv=nv).astype(np.float16)
                d[lo:hi] = buf
        tmp.replace(out)
        print(f"[done] {out}  n={n}  {time.time()-t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", default="1,5,9")
    ap.add_argument("--splits", default="train,val", help="NEVER include 'test' during a pilot")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--fs", type=int, default=200)
    ap.add_argument("--nv", type=int, default=16)
    ap.add_argument("--allow_test", action="store_true", help="explicit opt-in; off by default")
    a = ap.parse_args()
    splits = [s.strip() for s in a.splits.split(",")]
    if "test" in splits and not a.allow_test:
        raise SystemExit("Refusing to precompute the test split without --allow_test.")
    for f in [int(x) for x in a.folds.split(",")]:
        for s in splits:
            _precompute(DATA_ROOT / "loso" / f"fold_{f:02d}" / f"{s}.h5",
                        a.overwrite, a.batch_size, a.fs, a.nv)


if __name__ == "__main__":
    main()
