#!/usr/bin/env python3
"""Train a paper-faithful radarODE-MTL/EGA baseline on MMECG splits.

This script is intentionally separate from the earlier adapted radarODE run.
It follows the original radarODE-MTL training structure more closely:

  - input: WSST tensor, (B, 50, 71, 120)
  - tasks: ECG_shape, PPI, Anchor
  - outputs: ECG_shape=(B,1,200), PPI=(B,1,260), Anchor=(B,1,800)
  - multitask weighting: LibMTL EGA
  - optimizer defaults: SGD, lr=5e-3, momentum=0.937, wd=5e-4

The MMECG split protocol is kept compatible with the project's few-shot
calibration protocol: target subject 40 calibration segments, 10 validation
segments, and the remaining target segments for test.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[1]
RADARODE_ROOT = Path("/home/qhh2237/Projects/radarODE-MTL")
DATA_ROOT = Path("/home/qhh2237/Datasets/MMECG/processed")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RADARODE_ROOT))
sys.path.insert(0, str(RADARODE_ROOT / "Projects" / "radarODE_plus"))

from LibMTL.trainer import Trainer  # noqa: E402
from Projects.radarODE_plus.nets.PPI_decoder import PPI_decoder  # noqa: E402
from Projects.radarODE_plus.nets.anchor_decoder import anchor_decoder  # noqa: E402
from Projects.radarODE_plus.nets.backbone.dcnresnet_backbone import DCNResNet  # noqa: E402
from Projects.radarODE_plus.nets.backbone.squeeze_module import SqueezeModule  # noqa: E402
from Projects.radarODE_plus.utils.utils import (  # noqa: E402
    anchorLoss,
    anchorMetric,
    ppiLoss,
    ppiMetric,
    shapeLoss,
    shapeMetric,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def resample_1d(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == length:
        return x.astype(np.float32)
    return resample(x, length).astype(np.float32)


def split_calibration_indices(n: int, train_n: int, val_n: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    train_end = min(train_n, n)
    val_end = min(train_end + val_n, n)
    return indices[:train_end].tolist(), indices[train_end:val_end].tolist(), indices[val_end:].tolist()


def infer_subject_id_from_fold(fold_idx: int) -> str:
    # Existing MMECG LOSO folds are ordered by held-out subject.
    return f"S{fold_idx}"


class RadarODEBackbone(nn.Module):
    """Official radarODE feature extractor: DCNResNet + squeeze module."""

    def __init__(self, in_channels: int = 50) -> None:
        super().__init__()
        self.backbone = DCNResNet(in_channels)
        self.squeeze_module = SqueezeModule(in_channels=1024, out_channels=1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.squeeze_module(x)


class DeconvBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.batch_norm(self.conv_transpose(x)))


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.batch_norm(self.conv(x)))


class ShapeDecoder200(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            DeconvBlock1d(1024, 512, kernel_size=5, stride=3, padding=0),
            DeconvBlock1d(512, 256, kernel_size=5, stride=3, padding=0),
            DeconvBlock1d(256, 128, kernel_size=5, stride=3, padding=0),
        )
        self.decoder = nn.Sequential(
            ConvBlock1d(128, 64, kernel_size=7, stride=2, padding=1),
            ConvBlock1d(64, 32, kernel_size=7, stride=2, padding=1),
            ConvBlock1d(32, 16, kernel_size=7, stride=1, padding=1),
            ConvBlock1d(16, 8, kernel_size=5, stride=1, padding=1),
            nn.Conv1d(8, 4, kernel_size=7, stride=1, padding=1),
            nn.Conv1d(4, 2, kernel_size=5, stride=1, padding=1),
            nn.Conv1d(2, 1, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        out = self.encoder(rep)
        out = self.decoder(out)
        if out.shape[-1] != 200:
            out = F.interpolate(out, size=200, mode="linear", align_corners=False)
        return out


class PaperFaithfulMMECGDataset(Dataset):
    def __init__(self, h5_path: Path, wsst_path: Path) -> None:
        if not h5_path.exists():
            raise FileNotFoundError(h5_path)
        if not wsst_path.exists():
            raise FileNotFoundError(f"Missing WSST cache: {wsst_path}. Run scripts/precompute_wsst4s_mmecg.py first.")
        self.h5_path = h5_path
        self.wsst_path = wsst_path
        with h5py.File(h5_path, "r") as h5:
            self.n = int(h5["ecg"].shape[0])
        with h5py.File(wsst_path, "r") as h5:
            if tuple(h5["wsst"].shape[1:]) != (50, 71, 120):
                raise ValueError(f"{wsst_path} has wrong WSST shape: {h5['wsst'].shape}")
            if int(h5["wsst"].shape[0]) != self.n:
                raise ValueError(f"{wsst_path} sample count does not match {h5_path}")

    def __len__(self) -> int:
        return self.n

    def _get_rpeaks(self, h5: h5py.File, idx: int) -> np.ndarray:
        for key in ("rpeak_indices", "r_peak_indices", "rpeaks", "r_peaks", "peak_indices"):
            if key in h5:
                arr = np.asarray(h5[key][idx]).astype(np.int64)
                return arr[(arr >= 0) & (arr < 1600)]
        return np.empty((0,), dtype=np.int64)

    def _middle_cycle(self, ecg: np.ndarray, rpeaks: np.ndarray) -> np.ndarray:
        ecg = normalize_01(np.squeeze(ecg))
        rpeaks = np.sort(np.unique(rpeaks))
        if rpeaks.size >= 2:
            center = 800
            pairs = list(zip(rpeaks[:-1], rpeaks[1:]))
            containing = [(a, b) for a, b in pairs if a <= center <= b and b > a + 10]
            if containing:
                a, b = containing[0]
            else:
                a, b = min(pairs, key=lambda p: abs(((int(p[0]) + int(p[1])) / 2.0) - center))
            cycle = ecg[int(a) : int(b)]
            if cycle.size >= 40:
                return cycle.astype(np.float32)
        return ecg[700:900].astype(np.float32)

    def _anchor_mask(self, rpeaks: np.ndarray) -> np.ndarray:
        mask = np.zeros(800, dtype=np.float32)
        local = np.asarray(rpeaks, dtype=np.int64) - 400
        local = local[(local >= 0) & (local < 800)]
        if local.size == 0:
            return mask
        x = np.arange(800, dtype=np.float32)
        for peak in local:
            mask = np.maximum(mask, np.exp(-0.5 * ((x - float(peak)) / 4.0) ** 2).astype(np.float32))
        return mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with h5py.File(self.h5_path, "r") as h5, h5py.File(self.wsst_path, "r") as wh5:
            x = np.asarray(wh5["wsst"][idx], dtype=np.float32)
            ecg = np.asarray(h5["ecg"][idx], dtype=np.float32)
            rpeaks = self._get_rpeaks(h5, idx)

        cycle = self._middle_cycle(ecg, rpeaks)
        ecg_shape = normalize_01(resample_1d(cycle, 200))[None, :]

        if cycle.size > 260:
            ppi_cycle = normalize_01(resample_1d(cycle, 260))
        else:
            ppi_cycle = normalize_01(cycle)
        ppi = np.full(260, -10.0, dtype=np.float32)
        ppi[: min(ppi_cycle.size, 260)] = ppi_cycle[:260]

        anchor = self._anchor_mask(rpeaks)[None, :]

        labels = {
            "ECG_shape": torch.from_numpy(ecg_shape.astype(np.float32)),
            "PPI": torch.from_numpy(ppi[None, :].astype(np.float32)),
            "Anchor": torch.from_numpy(anchor.astype(np.float32)),
        }
        return torch.from_numpy(x.astype(np.float32)), labels


@dataclass
class SplitPaths:
    train: Path
    val: Path
    test: Path

    @property
    def train_wsst(self) -> Path:
        return self.train.with_name("train_wsst4s.h5")

    @property
    def val_wsst(self) -> Path:
        return self.val.with_name("val_wsst4s.h5")

    @property
    def test_wsst(self) -> Path:
        return self.test.with_name("test_wsst4s.h5")


def fold_paths(fold_idx: int) -> SplitPaths:
    fold_dir = DATA_ROOT / "loso" / f"fold_{fold_idx:02d}"
    return SplitPaths(fold_dir / "train.h5", fold_dir / "val.h5", fold_dir / "test.h5")


def build_loaders(args: argparse.Namespace) -> tuple[dict[str, DataLoader], dict[str, DataLoader], dict[str, DataLoader], dict[str, Any]]:
    paths = fold_paths(args.fold_idx)
    source_train = PaperFaithfulMMECGDataset(paths.train, paths.train_wsst)
    target_test_full = PaperFaithfulMMECGDataset(paths.test, paths.test_wsst)

    calib_idx, val_idx, test_idx = split_calibration_indices(
        len(target_test_full),
        train_n=args.calib_train_segments,
        val_n=args.calib_val_segments,
        seed=args.seed + args.fold_idx,
    )

    train_dataset: Dataset
    if args.protocol == "loso_calib":
        train_dataset = ConcatDataset([source_train, Subset(target_test_full, calib_idx)])
        val_dataset = Subset(target_test_full, val_idx)
        test_dataset = Subset(target_test_full, test_idx)
    elif args.protocol == "strict_loso":
        val_dataset_full = PaperFaithfulMMECGDataset(paths.val, paths.val_wsst)
        train_dataset = source_train
        val_dataset = val_dataset_full
        test_dataset = target_test_full
    else:
        raise ValueError(args.protocol)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    meta = {
        "protocol": args.protocol,
        "fold_idx": args.fold_idx,
        "held_out_subject": infer_subject_id_from_fold(args.fold_idx),
        "source_train_segments": len(source_train),
        "calibration_train_segments": len(calib_idx) if args.protocol == "loso_calib" else 0,
        "calibration_val_segments": len(val_idx) if args.protocol == "loso_calib" else 0,
        "test_segments": len(test_dataset),
        "input_shape": "(50,71,120)",
        "outputs": {"ECG_shape": "(1,200)", "PPI": "(1,260)", "Anchor": "(1,800)"},
    }
    return {"train": train_loader}, {"val": val_loader}, {"test": test_loader}, meta


def make_trainer(args: argparse.Namespace) -> Trainer:
    task_dict = {
        "ECG_shape": {"metrics": ["MSE"], "metrics_fn": shapeMetric(), "loss_fn": shapeLoss(), "weight": [0]},
        "PPI": {"metrics": ["PPI"], "metrics_fn": ppiMetric(), "loss_fn": ppiLoss(), "weight": [0]},
        "Anchor": {"metrics": ["CE"], "metrics_fn": anchorMetric(), "loss_fn": anchorLoss(), "weight": [0]},
    }
    decoders = nn.ModuleDict(
        {
            "ECG_shape": ShapeDecoder200(),
            "PPI": PPI_decoder(output_dim=260),
            "Anchor": anchor_decoder(),
        }
    )
    return Trainer(
        task_dict=task_dict,
        weighting="EGA",
        architecture="HPS",
        encoder_class=lambda: RadarODEBackbone(in_channels=50),
        decoders=decoders,
        rep_grad=False,
        multi_input=False,
        optim_param={"optim": "sgd", "lr": args.lr, "weight_decay": args.weight_decay, "momentum": args.momentum},
        scheduler_param={"scheduler": "cos", "eta_min": args.lr * 0.01, "T_max": args.t_max},
        save_path=str(args.output_dir),
        load_path=None,
        modelName=f"radarode_mtl_ega_fold{args.fold_idx:02d}",
        weight_args={"EGA_temp": args.ega_temp},
        arch_args={},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_tag", default="mmecg_radarode_mtl_ega_fewshot40v10")
    parser.add_argument("--protocol", choices=["loso_calib", "strict_loso"], default="loso_calib")
    parser.add_argument("--fold_idx", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=22)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib_train_segments", type=int, default=40)
    parser.add_argument("--calib_val_segments", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--t_max", type=int, default=100)
    parser.add_argument("--ega_temp", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir = ROOT / "experiments_mmecg" / args.exp_tag / f"fold_{args.fold_idx:02d}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_loaders, val_loaders, test_loaders, meta = build_loaders(args)
    meta.update(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "SGD",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "scheduler": "cosine",
            "EGA_temp": args.ega_temp,
            "note": "Paper-faithful radarODE-MTL/EGA training on MMECG 4s WSST inputs.",
        }
    )
    with (args.output_dir / "config.json").open("w") as f:
        json.dump(meta, f, indent=2)

    trainer = make_trainer(args)
    trainer.train(train_loaders, val_loaders, test_loaders, epochs=args.epochs)


if __name__ == "__main__":
    main()
