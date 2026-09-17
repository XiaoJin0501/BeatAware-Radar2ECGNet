"""Deduplicated subject-centric storage for full-window MMECG WSST tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


POOL_NAME = "wsst8s_subject_pool"
SUPPORT = "full_window_samples_0_1600"


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def window_key(subject_id: object, record_id: object, start_idx: object,
               end_idx: object) -> tuple[int, str, int, int]:
    return int(subject_id), _text(record_id), int(start_idx), int(end_idx)


@dataclass(frozen=True)
class PoolLocation:
    path: Path
    row: int


class SubjectWSSTPool:
    """Index subject files once and materialize split tensors in source order."""

    def __init__(self, pool_dir: str | Path):
        self.pool_dir = Path(pool_dir)
        files = sorted(self.pool_dir.glob("subject_*.h5"))
        if not files:
            raise FileNotFoundError(f"No subject pool files in {self.pool_dir}")
        self.index: dict[tuple[int, str, int, int], PoolLocation] = {}
        for path in files:
            with h5py.File(path, "r") as h:
                if h.attrs.get("support") != SUPPORT:
                    raise ValueError(f"Non-full-window pool file: {path}")
                n = len(h["wsst"])
                fields = (h["subject_id"][:], h["record_id"][:],
                          h["start_idx"][:], h["end_idx"][:])
                if any(len(x) != n for x in fields):
                    raise ValueError(f"Misaligned metadata in {path}")
                for row, values in enumerate(zip(*fields)):
                    key = window_key(*values)
                    if key in self.index:
                        raise ValueError(f"Duplicate pool key {key}")
                    self.index[key] = PoolLocation(path, row)

    def load_for_split(self, split_h5: str | Path) -> np.ndarray:
        """Return WSST in exactly the row order of a LOSO split H5."""
        with h5py.File(split_h5, "r") as h:
            keys = [window_key(*values) for values in zip(
                h["subject_id"][:], h["record_id"][:],
                h["start_idx"][:], h["end_idx"][:])]
        missing = [key for key in keys if key not in self.index]
        if missing:
            raise KeyError(f"Subject pool misses {len(missing)} windows; first={missing[0]}")

        grouped: dict[Path, list[tuple[int, int]]] = {}
        for output_row, key in enumerate(keys):
            loc = self.index[key]
            grouped.setdefault(loc.path, []).append((output_row, loc.row))

        result = None
        for path, pairs in grouped.items():
            # h5py fancy indices must be increasing. Read sorted, then scatter
            # back to the split's canonical row order.
            pairs.sort(key=lambda pair: pair[1])
            source_rows = np.asarray([p[1] for p in pairs], dtype=np.int64)
            with h5py.File(path, "r") as h:
                values = np.asarray(h["wsst"][source_rows])
            if result is None:
                result = np.empty((len(keys), *values.shape[1:]), dtype=values.dtype)
            for value, (output_row, _) in zip(values, pairs):
                result[output_row] = value
        if result is None:
            raise ValueError(f"Empty split: {split_h5}")
        return result
