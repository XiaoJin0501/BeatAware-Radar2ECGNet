#!/usr/bin/env python3
"""Audit the MMECG record-disjoint/subject-overlap protocol ladder arm."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path("/home/qhh2237/Datasets/MMECG/processed/samplewise")


def values(path: Path, key: str) -> set[str]:
    with h5py.File(path, "r") as handle:
        raw = np.asarray(handle[key])
    return {x.decode() if isinstance(x, bytes) else str(x) for x in raw}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    train = ROOT / "train.h5"
    val = ROOT / "val.h5"
    train_records, val_records = values(train, "record_id"), values(val, "record_id")
    train_subjects, val_subjects = values(train, "subject_id"), values(val, "subject_id")
    assert train_records.isdisjoint(val_records), "record leakage between train and validation"
    overlap = sorted(train_subjects & val_subjects, key=int)
    assert overlap, "this diagnostic requires subject overlap"
    with h5py.File(train, "r") as tr, h5py.File(val, "r") as va:
        result = {
            "protocol": "record-disjoint, subject-overlap validation",
            "test_opened": False,
            "train_windows": int(len(tr["ecg"])),
            "validation_windows": int(len(va["ecg"])),
            "train_records": len(train_records),
            "validation_records": len(val_records),
            "record_overlap": 0,
            "train_subjects": len(train_subjects),
            "validation_subjects": len(val_subjects),
            "overlap_subjects": overlap,
            "overlap_fraction_of_validation_subjects": len(overlap) / len(val_subjects),
            "train_sha256": sha256(train),
            "validation_sha256": sha256(val),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
