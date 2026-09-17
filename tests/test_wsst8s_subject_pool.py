from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from src.data.wsst8s_subject_pool import SUPPORT, SubjectWSSTPool


def _strings(values):
    return np.asarray(values, dtype=object)


def test_subject_pool_restores_split_order_bitwise(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    with h5py.File(pool / "subject_02.h5", "w") as h:
        h.attrs["support"] = SUPPORT
        h.create_dataset("wsst", data=np.asarray([[[[20]]], [[[21]]]], np.float16))
        h.create_dataset("subject_id", data=[2, 2])
        h.create_dataset("record_id", data=_strings(["r2", "r2"]),
                         dtype=h5py.string_dtype("utf-8"))
        h.create_dataset("start_idx", data=[0, 8])
        h.create_dataset("end_idx", data=[1600, 1608])
    with h5py.File(pool / "subject_01.h5", "w") as h:
        h.attrs["support"] = SUPPORT
        h.create_dataset("wsst", data=np.asarray([[[[10]]]], np.float16))
        h.create_dataset("subject_id", data=[1])
        h.create_dataset("record_id", data=_strings(["r1"]),
                         dtype=h5py.string_dtype("utf-8"))
        h.create_dataset("start_idx", data=[0])
        h.create_dataset("end_idx", data=[1600])
    split = tmp_path / "val.h5"
    with h5py.File(split, "w") as h:
        h.create_dataset("subject_id", data=[2, 1, 2])
        h.create_dataset("record_id", data=_strings(["r2", "r1", "r2"]),
                         dtype=h5py.string_dtype("utf-8"))
        h.create_dataset("start_idx", data=[8, 0, 0])
        h.create_dataset("end_idx", data=[1608, 1600, 1600])
    result = SubjectWSSTPool(pool).load_for_split(split)
    assert result.dtype == np.float16
    assert result.shape == (3, 1, 1, 1)
    assert result[:, 0, 0, 0].tolist() == [21.0, 10.0, 20.0]
