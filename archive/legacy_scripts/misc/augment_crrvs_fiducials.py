#!/usr/bin/env python3
"""Augment CR_RVS H5 splits with full-window ECG fiducial labels.

The first CR_RVS shape pipeline stored R-peaks but left Q/S/T arrays empty.
That is sufficient for Shape-200 morphology supervision, but it is not enough
for VMD-CBNeT-style timing diagnostics. This script delineates each full
8-second ECG window using the stored R-peaks and writes Q/S/T peak arrays back
to an MMECG-compatible H5 file.

The script deliberately treats these labels as ECG-derived ground truth. They
are valid for evaluation/supervision, but must not be used as radar-side input
or for test-time model selection.
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Any

import h5py
import neurokit2 as nk
import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/qhh2237/Datasets/CR_RVS/processed/q1_shape_kfold")
FS = 200


def _clean_indices(arr: Any, lo: int = 0, hi: int = 1600) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).reshape(-1)
    out = out[np.isfinite(out)].astype(np.int64)
    return np.sort(np.unique(out[(out >= lo) & (out < hi)]))


def _parse_wave_array(values: Any, hi: int = 1600) -> np.ndarray:
    vals = []
    for value in values:
        if value is None:
            continue
        try:
            if np.isnan(value):
                continue
        except TypeError:
            pass
        iv = int(round(float(value)))
        if 0 <= iv < hi:
            vals.append(iv)
    return np.asarray(sorted(set(vals)), dtype=np.int32)


def _delineate_window(ecg: np.ndarray, rpeaks: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    empty = np.asarray([], dtype=np.int32)
    rpeaks = _clean_indices(rpeaks, hi=len(ecg))
    if rpeaks.size < 2:
        return empty, empty, empty, 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, waves = nk.ecg_delineate(
                np.asarray(ecg, dtype=np.float64).reshape(-1),
                rpeaks.astype(int),
                sampling_rate=fs,
                method="dwt",
            )
        q = _parse_wave_array(waves.get("ECG_Q_Peaks", []), hi=len(ecg))
        s = _parse_wave_array(waves.get("ECG_S_Peaks", []), hi=len(ecg))
        t = _parse_wave_array(waves.get("ECG_T_Peaks", []), hi=len(ecg))
        valid = int(q.size > 0 and s.size > 0 and t.size > 0)
        return q, s, t, valid
    except Exception:
        return empty, empty, empty, 0


def _replace_vlen_i32(h5: h5py.File, key: str, arrays: list[np.ndarray]) -> None:
    if key in h5:
        del h5[key]
    ds = h5.create_dataset(key, shape=(len(arrays),), dtype=h5py.vlen_dtype(np.dtype("int32")))
    for idx, arr in enumerate(arrays):
        ds[idx] = np.asarray(arr, dtype=np.int32)


def _copy_if_needed(src: Path, dst: Path, overwrite: bool) -> Path:
    if src == dst:
        return src
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {dst}. Use --overwrite to replace it.")
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def augment_file(src_h5: Path, dst_h5: Path, overwrite: bool, fs: int) -> dict[str, Any]:
    path = _copy_if_needed(src_h5, dst_h5, overwrite=overwrite)
    q_arrays: list[np.ndarray] = []
    s_arrays: list[np.ndarray] = []
    t_arrays: list[np.ndarray] = []
    valid_flags: list[int] = []
    r_counts: list[int] = []
    q_counts: list[int] = []
    s_counts: list[int] = []
    t_counts: list[int] = []

    with h5py.File(path, "r+") as h5:
        n = int(h5["ecg"].shape[0])
        for idx in tqdm(range(n), desc=path.name, leave=False):
            ecg = np.asarray(h5["ecg"][idx], dtype=np.float32).reshape(-1)
            r = _clean_indices(h5["rpeak_indices"][idx], hi=len(ecg))
            q, s, t, valid = _delineate_window(ecg, r, fs=fs)
            q_arrays.append(q)
            s_arrays.append(s)
            t_arrays.append(t)
            valid_flags.append(valid)
            r_counts.append(int(r.size))
            q_counts.append(int(q.size))
            s_counts.append(int(s.size))
            t_counts.append(int(t.size))

        _replace_vlen_i32(h5, "q_indices", q_arrays)
        _replace_vlen_i32(h5, "s_indices", s_arrays)
        _replace_vlen_i32(h5, "tpeak_indices", t_arrays)
        if "delineation_valid" in h5:
            del h5["delineation_valid"]
        h5.create_dataset("delineation_valid", data=np.asarray(valid_flags, dtype=np.uint8))

    valid_arr = np.asarray(valid_flags, dtype=bool)
    return {
        "path": str(path),
        "num_segments": int(len(valid_flags)),
        "valid_rate": float(valid_arr.mean()) if len(valid_arr) else float("nan"),
        "r_count_mean": float(np.mean(r_counts)) if r_counts else float("nan"),
        "q_count_mean": float(np.mean(q_counts)) if q_counts else float("nan"),
        "s_count_mean": float(np.mean(s_counts)) if s_counts else float("nan"),
        "t_count_mean": float(np.mean(t_counts)) if t_counts else float("nan"),
        "segments_with_q": int(np.sum(np.asarray(q_counts) > 0)),
        "segments_with_s": int(np.sum(np.asarray(s_counts) > 0)),
        "segments_with_t": int(np.sum(np.asarray(t_counts) > 0)),
    }


def _fold_split_path(root: Path, fold: int, split: str) -> Path:
    return root / f"fold_{fold:02d}" / f"{split}.h5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    parser.add_argument("--fs", type=int, default=FS)
    parser.add_argument("--in_place", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--audit_dir",
        type=Path,
        default=ROOT / "experiments_mmecg" / "q1_crrvs_fiducial_audit",
    )
    args = parser.parse_args()

    if not args.in_place and args.output_root is None:
        raise ValueError("Use --in_place or provide --output_root.")
    if args.in_place:
        args.output_root = args.data_root

    args.audit_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in args.folds:
        for split in args.splits:
            src = _fold_split_path(args.data_root, fold, split)
            dst = _fold_split_path(args.output_root, fold, split)
            if not src.exists():
                raise FileNotFoundError(src)
            row = augment_file(src, dst, overwrite=args.overwrite, fs=args.fs)
            row.update({"fold": int(fold), "split": split, "fs": int(args.fs)})
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.audit_dir / "crrvs_fiducial_augmentation_audit.csv", index=False)
    summary = {
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "in_place": bool(args.in_place),
        "folds": [int(f) for f in args.folds],
        "splits": list(args.splits),
        "mean_valid_rate": float(df["valid_rate"].mean()) if len(df) else float("nan"),
        "audit_csv": str(args.audit_dir / "crrvs_fiducial_augmentation_audit.csv"),
        "note": (
            "Q/S/T labels are ECG-derived full-window fiducials. They enable timing "
            "diagnostics but are not radar inputs and must not be used for test-time "
            "selection."
        ),
    }
    (args.audit_dir / "crrvs_fiducial_augmentation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
