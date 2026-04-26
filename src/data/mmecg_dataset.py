"""
mmecg_dataset.py — MMECGDataset

加载预处理后的 MMECG NPY 数据，支持 LOSO (Leave-One-Subject-Out) 划分。

Usage:
    from src.data.mmecg_dataset import MMECGDataset, build_loso_loaders

    train_loader, test_loader = build_loso_loaders(
        dataset_dir="dataset_mmecg",
        fold_idx=0,          # 0~10，留出第 fold_idx 号受试者作测试
        batch_size=16,
    )
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class MMECGDataset(Dataset):
    """
    Parameters
    ----------
    dataset_dir : str | Path
        预处理输出目录（含 subject_*/rcg.npy 等）
    subject_ids : list[int]
        参与本 split 的受试者 ID 列表
    """

    def __init__(self, dataset_dir: str | Path, subject_ids: list[int]):
        super().__init__()
        self.dataset_dir  = Path(dataset_dir)
        self.subject_ids  = subject_ids

        rcg_list, ecg_list, rpeak_list, meta_list = [], [], [], []

        for sid in subject_ids:
            subj_dir = self.dataset_dir / f"subject_{sid}"
            rcg_arr   = np.load(subj_dir / "rcg.npy")    # (N, 50, 1600)
            ecg_arr   = np.load(subj_dir / "ecg.npy")    # (N,  1, 1600)
            rp_arr    = np.load(subj_dir / "rpeak.npy")  # (N,  1, 1600)
            meta_arr  = np.load(subj_dir / "meta.npy")   # (N,  2)  int32

            rcg_list.append(rcg_arr)
            ecg_list.append(ecg_arr)
            rpeak_list.append(rp_arr)
            meta_list.append(meta_arr)

        self.rcg   = np.concatenate(rcg_list,   axis=0)   # (N_total, 50, 1600)
        self.ecg   = np.concatenate(ecg_list,   axis=0)   # (N_total,  1, 1600)
        self.rpeak = np.concatenate(rpeak_list, axis=0)   # (N_total,  1, 1600)
        self.meta  = np.concatenate(meta_list,  axis=0)   # (N_total,  2)

    def __len__(self) -> int:
        return len(self.rcg)

    def __getitem__(self, idx: int) -> dict:
        return {
            "radar":   torch.from_numpy(self.rcg[idx]),    # (50, 1600)
            "ecg":     torch.from_numpy(self.ecg[idx]),    # (1,  1600)
            "rpeak":   torch.from_numpy(self.rpeak[idx]),  # (1,  1600)
            "subject": int(self.meta[idx, 0]),
            "state":   int(self.meta[idx, 1]),
        }

    def subject_weights(self) -> torch.Tensor:
        """
        计算每个样本的权重，使训练时各受试者贡献均衡
        （WeightedRandomSampler 用）。
        """
        subj_ids = self.meta[:, 0]
        unique, counts = np.unique(subj_ids, return_counts=True)
        inv_freq = 1.0 / counts.astype(np.float32)
        subj2weight = dict(zip(unique.tolist(), inv_freq.tolist()))
        weights = np.array([subj2weight[s] for s in subj_ids], dtype=np.float32)
        return torch.from_numpy(weights)


def build_loso_loaders(
    dataset_dir: str | Path,
    fold_idx: int,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    balanced_sampling: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    构建 LOSO 的训练 / 测试 DataLoader。

    Parameters
    ----------
    fold_idx : int
        0-based fold 序号（= LOSO 中留出的受试者在有序列表中的下标）。
    balanced_sampling : bool
        True → 用 WeightedRandomSampler 对受试者均衡采样；
        False → 普通随机 shuffle。

    Returns
    -------
    train_loader, test_loader
    """
    meta_path = Path(dataset_dir) / "metadata_mmecg.json"
    with open(meta_path) as f:
        meta = json.load(f)

    fold_info = meta["loso_folds"][str(fold_idx)]
    test_sid  = [fold_info["test"]]
    train_sids = fold_info["train"]

    train_ds = MMECGDataset(dataset_dir, train_sids)
    test_ds  = MMECGDataset(dataset_dir, test_sid)

    if balanced_sampling:
        weights = train_ds.subject_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
