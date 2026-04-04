"""
dataset.py — RadarECGDataset

支持：
  - 三种雷达输入表征：raw / phase / spec
  - 5-Fold Cross-Validation（按受试者划分）
  - 多场景：resting / valsalva / apnea
  - 惰性加载（mmap_mode='r'，不预加载到内存）
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


InputType = Literal["raw", "phase", "spec"]
Split     = Literal["train", "val"]


class RadarECGDataset(Dataset):
    """
    从预处理好的 NPY 分段文件中加载数据。

    Parameters
    ----------
    dataset_dir : Path
        数据集根目录（含 metadata.json 和各受试者子目录）
    fold_idx : int
        当前 fold 索引（0~4）。val 使用该 fold，train 使用其余所有 fold。
    split : 'train' | 'val'
    input_type : 'raw' | 'phase' | 'spec'
        雷达输入表征类型
    scenarios : list[str] | None
        使用的场景列表，默认 ['resting', 'valsalva', 'apnea']

    Returns (per __getitem__)
    -------------------------
    dict:
        'radar'   : Tensor [1, 1600] (raw/phase) 或 [1, 33, 196] (spec)
        'ecg'     : Tensor [1, 1600]
        'rpeak'   : Tensor [1, 1600]
        'subject' : str
        'scenario': str
    """

    _RADAR_FILE = {
        "raw":   "radar_raw.npy",
        "phase": "radar_phase.npy",
        "spec":  "radar_spec_input.npy",
    }

    def __init__(
        self,
        dataset_dir:  Path | str,
        fold_idx:     int,
        split:        Split,
        input_type:   InputType = "phase",
        scenarios:    list[str] | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.input_type  = input_type
        self.scenarios   = scenarios or ["resting", "valsalva", "apnea"]

        # ── 读取 metadata ─────────────────────────────────────────────────
        meta_path = self.dataset_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json 不存在: {meta_path}")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        folds: dict[str, list[str]] = meta["fold_assignments"]["folds"]
        val_key = f"fold_{fold_idx}"
        if val_key not in folds:
            raise ValueError(f"fold_idx={fold_idx} 不存在，可用: {list(folds.keys())}")

        val_subjects   = set(folds[val_key])
        all_subjects   = set(meta["final_subjects"])
        train_subjects = all_subjects - val_subjects

        subjects = val_subjects if split == "val" else train_subjects

        # ── 建立分段索引表 ────────────────────────────────────────────────
        # 每个 entry: (radar_path, ecg_path, rpeak_path, row_idx, subject, scenario)
        self._index: list[tuple] = []

        radar_fname = self._RADAR_FILE[input_type]

        for subject in sorted(subjects):
            for scenario in self.scenarios:
                seg_dir = self.dataset_dir / subject / scenario / "segments"
                radar_path = seg_dir / radar_fname
                ecg_path   = seg_dir / "ecg.npy"
                rpeak_path = seg_dir / "rpeak.npy"

                if not (radar_path.exists() and ecg_path.exists()
                        and rpeak_path.exists()):
                    continue

                n = np.load(ecg_path, mmap_mode="r").shape[0]
                for i in range(n):
                    self._index.append(
                        (radar_path, ecg_path, rpeak_path, i, subject, scenario)
                    )

        if len(self._index) == 0:
            raise RuntimeError(
                f"数据集为空（fold={fold_idx}, split={split}, "
                f"input_type={input_type}）。请检查 dataset_dir 和 metadata.json。"
            )

        # ── mmap 缓存（避免每次 __getitem__ 重新打开文件）────────────────
        self._mmap_cache: dict[str, np.ndarray] = {}

    def _load_mmap(self, path: Path) -> np.ndarray:
        key = str(path)
        if key not in self._mmap_cache:
            self._mmap_cache[key] = np.load(path, mmap_mode="r")
        return self._mmap_cache[key]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        radar_path, ecg_path, rpeak_path, row, subject, scenario = self._index[idx]

        radar = self._load_mmap(radar_path)[row]    # [1, 1600] or [1, 33, 200]
        ecg   = self._load_mmap(ecg_path)[row]      # [1, 1600]
        rpeak = self._load_mmap(rpeak_path)[row]    # [1, 1600]

        return {
            "radar":    torch.from_numpy(radar.copy()).float(),
            "ecg":      torch.from_numpy(ecg.copy()).float(),
            "rpeak":    torch.from_numpy(rpeak.copy()).float(),
            "subject":  subject,
            "scenario": scenario,
        }

    def __repr__(self) -> str:
        return (
            f"RadarECGDataset("
            f"n={len(self)}, input_type={self.input_type}, "
            f"scenarios={self.scenarios})"
        )
