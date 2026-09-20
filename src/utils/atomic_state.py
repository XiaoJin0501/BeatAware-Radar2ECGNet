"""Atomic artifact writes and reproducible training-state capture."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch


def _replace(tmp: Path, target: Path) -> None:
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, target)


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_text(path: str | Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    _replace(tmp, path)


def atomic_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_csv(path: str | Path, fieldnames: list[str],
               rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def capture_rng_state(train_generator: torch.Generator | None = None) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "train_generator": train_generator.get_state() if train_generator is not None else None,
    }


def _byte_state(t: object) -> torch.Tensor:
    """Torch RNG states must be CPU ByteTensors.

    A checkpoint loaded with `map_location="cuda"` brings them back on the GPU, and
    `set_rng_state` then raises `RNG state must be a torch.ByteTensor`. Moving them
    back is what makes an interrupted GPU run resumable.
    """
    return torch.as_tensor(t, dtype=torch.uint8).cpu()


def restore_rng_state(state: Mapping[str, object],
                      train_generator: torch.Generator | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_byte_state(state["torch_cpu"]))
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([_byte_state(t) for t in state["torch_cuda"]])
    if train_generator is not None and state.get("train_generator") is not None:
        train_generator.set_state(_byte_state(state["train_generator"]))


STAGES = ("TRAINING", "CHECKPOINT_FROZEN", "TEST_EVALUATING", "COMPLETE")


def transition_state(path: str | Path, new_stage: str, **metadata: object) -> dict:
    path = Path(path)
    if new_stage not in STAGES:
        raise ValueError(f"Unknown stage {new_stage}")
    old = json.loads(path.read_text()) if path.exists() else None
    if old is None:
        if new_stage != "TRAINING":
            raise RuntimeError(f"First stage must be TRAINING, not {new_stage}")
    else:
        old_stage = old["stage"]
        if old_stage == "COMPLETE" and new_stage != "COMPLETE":
            raise RuntimeError("COMPLETE is terminal; test/training is permanently sealed")
        if STAGES.index(new_stage) < STAGES.index(old_stage):
            raise RuntimeError(f"Illegal state regression {old_stage} -> {new_stage}")
        if STAGES.index(new_stage) > STAGES.index(old_stage) + 1:
            raise RuntimeError(f"Illegal state jump {old_stage} -> {new_stage}")
    record = {"stage": new_stage, **metadata}
    atomic_json(path, record)
    return record
