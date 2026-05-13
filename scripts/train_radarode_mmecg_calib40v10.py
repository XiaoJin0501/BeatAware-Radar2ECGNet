"""
train_radarode_mmecg_calib40v10.py

radarODE-MTL backbone + shapeDecoder adapted for MMECG calib40v10 protocol.
For use as a paper comparison baseline.

Architecture faithfully follows radarODE-MTL (radarODE+, AAAI 2024):
  DCNResNet (in_ch=50) → SqueezeModule → LSTMCNNEncoder → CNNDecoder
  Input:  (B, 50, 71, 120) WSST magnitude spectrogram (or STFT fallback)
  Output: (B, 1, 1600)  ECG in [0, 1]  (original outputs 200pt; upsampled here)

Key adaptations for MMECG:
  1. WSST preprocessing (preferred, matches original radarODE-MTL MATLAB pipeline):
       RCG (50, 1600) resampled 200→30 Hz → ssqueezepy ssq_cwt (Morlet, nv=10)
       → (72, 240), take [:71, ::2] → (71, 120), normalised per-spectrogram to [0,1]
       Pre-computed by scripts/precompute_wsst_mmecg.py → *_wsst.h5 companion files.
       Falls back to STFT if companion not found.
  2. Decoder output F.interpolated 200 → 1600 points
  3. Data: MMECGWindowedH5Dataset with loso_calib calib40v10 split
  4. Evaluation: BeatAware 4-level metrics for a fair comparison

Training hyperparams follow radarODE-MTL paper:
  SGD lr=5e-3, momentum=0.9, cosine annealing, 200 epochs
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import scipy.signal as sg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    WeightedRandomSampler,
)

ROOT = Path(__file__).resolve().parent.parent
RADARODE_ROOT = Path("/home/qhh2237/Projects/radarODE-MTL")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RADARODE_ROOT))
sys.path.insert(0, str(RADARODE_ROOT / "Projects/radarODE_plus"))

from scripts.test_mmecg import _evaluate_segment, _json_default, _loso_summary
from src.data.mmecg_dataset import (
    MMECGWindowedH5Dataset,
    _split_calibration_indices,
)
from src.utils.metrics import summarize_global_metrics, summarize_subject_metrics

from Projects.radarODE_plus.nets.PPI_decoder import PPI_decoder
from Projects.radarODE_plus.nets.anchor_decoder import anchor_decoder


# ─────────────────────────────────────────────────────────────────
#  STFT preprocessing  (50, 1600) → (50, 71, 120)
# ─────────────────────────────────────────────────────────────────

def rcg_to_stft(rcg: np.ndarray) -> np.ndarray:
    """(R, L) float32 → (R, 71, 120) STFT magnitude, normalised to [0, 1].

    Fallback when WSST companion H5 is not available.
    """
    specs: list[np.ndarray] = []
    for b in range(rcg.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, Zxx = sg.stft(
                rcg[b], fs=200, window="hann", nperseg=140, noverlap=128
            )
        mag = np.abs(Zxx)[:71, :120]      # (71, T_trim) → (71, 120)
        if mag.shape[1] < 120:
            mag = np.pad(mag, ((0, 0), (0, 120 - mag.shape[1])))
        max_val = float(mag.max())
        if max_val > 1e-8:
            mag = mag / max_val
        specs.append(mag)
    return np.stack(specs, axis=0).astype(np.float32)  # (R, 71, 120)


def rcg_to_wsst(rcg: np.ndarray) -> np.ndarray:
    """(R, L) float32 → (R, 71, 120) WSST magnitude [0, 1].

    Replicates MATLAB wsst(x, 200, 'VoicesPerOctave', 10) from MMECG_to_SST.m:
      1. Resample 200 Hz → 30 Hz (240 pts for 8 s)
      2. ssq_cwt with Morlet wavelet, nv=10 → (72, 240)
      3. Take [:71, ::2] → (71, 120), normalise per-spectrogram
    """
    from ssqueezepy import ssq_cwt as _ssq_cwt
    specs: list[np.ndarray] = []
    for b in range(rcg.shape[0]):
        x_30 = sg.resample(rcg[b].astype(np.float64), 240)
        Tx, _, _, _ = _ssq_cwt(x_30, wavelet="morlet", fs=30, nv=10)
        mag = np.abs(Tx[:71, ::2]).astype(np.float32)   # (71, 120)
        vmax = float(mag.max())
        if vmax > 1e-8:
            mag /= vmax
        specs.append(mag)
    return np.stack(specs, axis=0)   # (R, 71, 120)


# Companion WSST H5 suffix written by precompute_wsst_mmecg.py
_WSST_SUFFIX = "_wsst.h5"


def _load_wsst_h5(h5_path: Path) -> "np.ndarray | None":
    """Return wsst array (N, 50, 71, 120) float32 from companion *_wsst.h5, or None."""
    wsst_path = h5_path.with_name(h5_path.stem + _WSST_SUFFIX)
    if not wsst_path.exists():
        return None
    with h5py.File(wsst_path, "r") as f:
        return f["wsst"][:].astype(np.float32)  # stored as float16, load as float32


# ─────────────────────────────────────────────────────────────────
#  radarODE-MTL Architecture (inline, adapted)
#  Source: radarODE-MTL / Projects/radarODE_plus/nets/
# ─────────────────────────────────────────────────────────────────

class _DeformConv2d(nn.Module):
    """Pure-PyTorch deformable convolution (copied from radarODE-MTL dcnv2.py)."""

    def __init__(self, inc: int, outc: int, kernel_size: int = 3,
                 padding: int = 1, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size,
                              stride=kernel_size)
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size,
                                kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)  # type: ignore[arg-type]

    @staticmethod
    def _set_lr(module, grad_input, grad_output):  # noqa: ANN001
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.p_conv(x)
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2
        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([
            torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
            torch.clamp(q_lt[..., N:], 0, x.size(3) - 1),
        ], dim=-1).long()
        q_rb = torch.cat([
            torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
            torch.clamp(q_rb[..., N:], 0, x.size(3) - 1),
        ], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        p = torch.cat([
            torch.clamp(p[..., :N], 0, x.size(2) - 1),
            torch.clamp(p[..., N:], 0, x.size(3) - 1),
        ], dim=-1)
        g_lt = ((1 + (q_lt[..., :N].type_as(p) - p[..., :N])) *
                (1 + (q_lt[..., N:].type_as(p) - p[..., N:])))
        g_rb = ((1 - (q_rb[..., :N].type_as(p) - p[..., :N])) *
                (1 - (q_rb[..., N:].type_as(p) - p[..., N:])))
        g_lb = ((1 + (q_lb[..., :N].type_as(p) - p[..., :N])) *
                (1 - (q_lb[..., N:].type_as(p) - p[..., N:])))
        g_rt = ((1 - (q_rt[..., :N].type_as(p) - p[..., :N])) *
                (1 + (q_rt[..., N:].type_as(p) - p[..., N:])))
        xqlt = self._get_x_q(x, q_lt, N)
        xqrb = self._get_x_q(x, q_rb, N)
        xqlb = self._get_x_q(x, q_lb, N)
        xqrt = self._get_x_q(x, q_rt, N)
        x_off = (g_lt.unsqueeze(1) * xqlt + g_rb.unsqueeze(1) * xqrb +
                 g_lb.unsqueeze(1) * xqlb + g_rt.unsqueeze(1) * xqrt)
        x_off = self._reshape_x_offset(x_off, ks)
        return self.conv(x_off)

    def _get_p_n(self, N: int, dtype: str) -> torch.Tensor:
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2,
                         (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2,
                         (self.kernel_size - 1) // 2 + 1),
            indexing="ij",
        )
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        return p_n.view(1, 2 * N, 1, 1).type(dtype)

    def _get_p_0(self, h: int, w: int, N: int,
                 dtype: str) -> torch.Tensor:
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
            indexing="ij",
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        return torch.cat([p_0_x, p_0_y], 1).type(dtype)

    def _get_p(self, offset: torch.Tensor, dtype: str) -> torch.Tensor:
        N = offset.size(1) // 2
        h, w = offset.size(2), offset.size(3)
        return self._get_p_0(h, w, N, dtype) + self._get_p_n(N, dtype) + offset

    def _get_x_q(self, x: torch.Tensor, q: torch.Tensor,
                 N: int) -> torch.Tensor:
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = (index.contiguous().unsqueeze(1)
                 .expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1))
        return x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

    @staticmethod
    def _reshape_x_offset(x_off: torch.Tensor, ks: int) -> torch.Tensor:
        b, c, h, w, N = x_off.size()
        x_off = torch.cat(
            [x_off[..., s:s + ks].contiguous().view(b, c, h, w * ks)
             for s in range(0, N, ks)],
            dim=-1,
        )
        return x_off.contiguous().view(b, c, h * ks, w * ks)


class _BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size, stride, padding):
        super().__init__()
        self.dcn1 = _DeformConv2d(in_ch, in_ch, kernel_size=3,
                                   padding=1, stride=1)
        self.bn = nn.BatchNorm2d(in_ch)
        self.act = nn.ReLU(inplace=False)
        self.dcn2 = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn(self.dcn1(x))) + x
        return self.dcn2(out)


class _Downsampling(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _DCNResNet(nn.Module):
    """DCN-ResNet backbone (radarODE-MTL, in_ch=50).
    Input: (B, 50, 71, 120) → Output: (B, 1024, 6, 31)
    """
    def __init__(self, in_ch: int = 50):
        super().__init__()
        ch = [in_ch, 128, 256, 512, 1024]
        self.stage0 = _BasicBlock(ch[0], ch[1], (2, 1), (1, 1), (1, 0))
        self.down0  = _Downsampling(ch[1], ch[1], (3, 2), 2, 1)
        self.stage1 = _BasicBlock(ch[1], ch[2], (2, 1), (1, 1), (1, 0))
        self.down1  = _Downsampling(ch[2], ch[2], (3, 2), 2, 1)
        self.stage2 = _BasicBlock(ch[2], ch[3], (2, 1), (1, 1), (1, 0))
        self.down2  = _Downsampling(ch[3], ch[3], (3, 3), (2, 1), 1)
        self.stage3 = _BasicBlock(ch[3], ch[4], (2, 1), (1, 1), (1, 0))
        self.down3  = _Downsampling(ch[4], ch[4], (3, 3), (2, 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down0(self.stage0(x))
        x = self.down1(self.stage1(x))
        x = self.down2(self.stage2(x))
        x = self.down3(self.stage3(x))
        return x


class _SqueezeModule(nn.Module):
    """Squeeze spatial H dimension to 1 (radarODE-MTL squeeze_module.py)."""
    def __init__(self, in_ch: int = 1024, out_ch: int = 1024,
                 squeeze_h: int = 6):
        super().__init__()
        self.squeeze = nn.Conv2d(in_ch, out_ch,
                                 kernel_size=(squeeze_h, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.squeeze(x).squeeze(2)  # (B, C, W)


class _DeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv_t = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv_t(x)))


class _LSTMCNNEncoder(nn.Module):
    """LSTMCNNEncoder (radarODE-MTL encoder.py, deconv path only).
    Input: (B, 1024, 31) → Output: (B, 128, 863)
    """
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.deconv = nn.Sequential(
            _DeconvBlock(dim, dim // 2, 5, 3, 0),
            _DeconvBlock(dim // 2, dim // 4, 5, 3, 0),
            _DeconvBlock(dim // 4, dim // 8, 5, 3, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class _ConvBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _CNNDecoder(nn.Module):
    """CNNLSTMDecoder temporal path (radarODE-MTL decoder.py, no ODE/fusion).
    Input: (B, 128, 863) → Output: (B, 1, ~200)
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBlock1d(128, 64, 7, 2, 1),
            _ConvBlock1d(64, 32, 7, 2, 1),
            _ConvBlock1d(32, 16, 7, 1, 1),
            _ConvBlock1d(16, 8, 5, 1, 1),
        )
        self.out_conv = nn.Sequential(
            nn.Conv1d(8, 4, 7, 1, 1),
            nn.Conv1d(4, 2, 5, 1, 1),
            nn.Conv1d(2, 1, 2, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_conv(self.conv(x))


class RadarODEMTLModel(nn.Module):
    """
    radarODE-MTL backbone + three task heads adapted for MMECG.

    Tasks:
      - ECG_shape: CNNLSTMDecoder waveform reconstruction
      - PPI: 260-way pulse interval/beat-count auxiliary classifier
      - Anchor: 800-point R-peak anchor auxiliary regression

    The ECG_shape output is interpolated from the original ~200-point decoder
    output to 1600 for MMECG's evaluation length.
    """

    def __init__(self):
        super().__init__()
        self.backbone = _DCNResNet(in_ch=50)
        self.squeeze  = _SqueezeModule(in_ch=1024, out_ch=1024, squeeze_h=6)
        self.encoder  = _LSTMCNNEncoder(dim=1024)
        self.decoder  = _CNNDecoder()
        self.ppi_decoder = PPI_decoder(output_dim=260)
        self.anchor_decoder = anchor_decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 50, 71, 120) → dict of task predictions."""
        feat = self.squeeze(self.backbone(x))        # (B, 1024, 31)
        ppi = self.ppi_decoder(feat.unsqueeze(2))    # (B, 1, 260)
        anchor = self.anchor_decoder(feat)           # (B, 1, 800)
        shape_feat = self.encoder(feat)              # (B, 128, 863)
        ecg = self.decoder(shape_feat)               # (B, 1, ~200)
        ecg = F.interpolate(ecg, size=1600,
                            mode="linear", align_corners=False)
        return {
            "ECG_shape": torch.sigmoid(ecg),
            "PPI": ppi,
            "Anchor": anchor,
        }


# ─────────────────────────────────────────────────────────────────
#  Dataset wrapper: STFT on-the-fly
# ─────────────────────────────────────────────────────────────────

class RadarODEDataset(Dataset):
    """Wraps MMECGWindowedH5Dataset, provides WSST (preferred) or STFT spectrogram.

    If wsst_cache is provided (precomputed, shape (N, 50, 71, 120)), uses it
    directly for fast I/O.  Falls back to on-the-fly STFT otherwise.
    """

    def __init__(self, base: Dataset,
                 include_peak_indices: bool = False,
                 wsst_cache: "np.ndarray | None" = None):
        self.base = base
        self.include_peak_indices = include_peak_indices
        self.wsst_cache = wsst_cache   # (N, 50, 71, 120) float32 or None

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        if self.wsst_cache is not None:
            spec = self.wsst_cache[idx]              # (50, 71, 120)
        else:
            rcg_np = item["radar"].numpy()           # (50, 1600)
            spec = rcg_to_stft(rcg_np)               # fallback
        out: dict = {
            "stft":    torch.from_numpy(spec),       # (50, 71, 120)
            "ecg":     item["ecg"].float(),           # (1, 1600) in [0, 1]
            "rpeak":   item["rpeak"].float(),         # (1, 1600) soft R mask
            "subject": item["subject"],
            "state":   item["state"],
        }
        for key in ("r_idx", "q_idx", "s_idx", "t_idx", "delin_valid"):
            if key in item:
                out[key] = item[key]
        return out


def collate_radarode(batch: list[dict]) -> dict:
    out: dict = {}
    for key in ("stft", "ecg", "rpeak"):
        out[key] = torch.stack([b[key] for b in batch])
    for key in ("subject", "state"):
        out[key] = torch.tensor([b[key] for b in batch], dtype=torch.long)
    for key in ("r_idx", "q_idx", "s_idx", "t_idx"):
        if key in batch[0]:
            out[key] = [b[key] for b in batch]
    if "delin_valid" in batch[0]:
        out["delin_valid"] = torch.tensor(
            [bool(b["delin_valid"]) for b in batch], dtype=torch.bool
        )
    return out


# ─────────────────────────────────────────────────────────────────
#  Subject-balanced sampler for base datasets
# ─────────────────────────────────────────────────────────────────

def _collect_subjects(ds: Dataset) -> np.ndarray:
    if isinstance(ds, MMECGWindowedH5Dataset):
        return ds._subj
    if isinstance(ds, Subset):
        return _collect_subjects(ds.dataset)[np.asarray(ds.indices)]
    if isinstance(ds, ConcatDataset):
        return np.concatenate(
            [_collect_subjects(child) for child in ds.datasets]
        )
    raise TypeError(f"Cannot extract subjects from {type(ds)!r}")


def _subject_sampler(train_data: Dataset) -> WeightedRandomSampler:
    subjects = _collect_subjects(train_data)
    unique, counts = np.unique(subjects, return_counts=True)
    inv = 1.0 / counts.astype(np.float32)
    w_map = dict(zip(unique.tolist(), inv.tolist()))
    weights = torch.tensor(
        [w_map[int(s)] for s in subjects], dtype=torch.float32
    )
    return WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True
    )


# ─────────────────────────────────────────────────────────────────
#  DataLoader builders
# ─────────────────────────────────────────────────────────────────

LOSO_H5_DIR = "/home/qhh2237/Datasets/MMECG/processed/loso"
DS_KWARGS = dict(
    narrow_bandpass=False,
    target_norm="minmax",
    topk_bins=None,
    topk_method="energy",
)


def _wsst_for_indices(wsst_arr: "np.ndarray | None",
                      indices: np.ndarray) -> "np.ndarray | None":
    """Select rows from a WSST array by index; return None if array is None."""
    return wsst_arr[indices] if wsst_arr is not None else None


def build_loaders(
    fold: int,
    batch_size: int,
    num_workers: int,
    protocol: str = "loso_calib",
    calib_n_train: int = 40,
    calib_n_val: int = 10,
    calib_seed: int = 42,
    balanced_sampling: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    fold_dir = Path(LOSO_H5_DIR) / f"fold_{fold:02d}"
    train_base = MMECGWindowedH5Dataset(
        fold_dir / "train.h5", include_peak_indices=True, **DS_KWARGS
    )
    # Load pre-computed WSST companion arrays (None if not yet precomputed)
    train_wsst = _load_wsst_h5(fold_dir / "train.h5")
    test_wsst  = _load_wsst_h5(fold_dir / "test.h5")
    using_wsst = (train_wsst is not None) or (test_wsst is not None)
    if using_wsst:
        print(f"  [WSST] Loaded companion .h5 files for fold {fold:02d} "
              f"(train={'OK' if train_wsst is not None else 'missing'}, "
              f"test={'OK' if test_wsst is not None else 'missing'})")
    else:
        print(f"  [STFT fallback] No WSST companion .h5 found for fold {fold:02d}")

    if protocol == "loso_calib":
        test_plain = MMECGWindowedH5Dataset(
            fold_dir / "test.h5", include_peak_indices=True, **DS_KWARGS
        )
        test_full = MMECGWindowedH5Dataset(
            fold_dir / "test.h5", include_peak_indices=True, **DS_KWARGS
        )
        calib_idx, calib_val_idx, eval_idx = _split_calibration_indices(
            test_plain._state,
            calib_ratio=0.4,
            calib_val_ratio=0.1,
            calib_n_train=calib_n_train,
            calib_n_val=calib_n_val,
            seed=calib_seed + fold,
        )
        calib_data = Subset(test_plain, calib_idx.tolist())
        train_data = ConcatDataset([train_base, calib_data])
        val_data   = Subset(test_plain, calib_val_idx.tolist())
        test_data  = Subset(test_full, eval_idx.tolist())

        # Build WSST caches aligned with train/val/test data
        if train_wsst is not None and test_wsst is not None:
            calib_wsst = _wsst_for_indices(test_wsst, calib_idx)
            train_wsst_full = np.concatenate([train_wsst, calib_wsst], axis=0)
            val_wsst_arr  = _wsst_for_indices(test_wsst, calib_val_idx)
            test_wsst_arr = _wsst_for_indices(test_wsst, eval_idx)
        else:
            train_wsst_full = val_wsst_arr = test_wsst_arr = None

    elif protocol == "loso":
        val_base  = MMECGWindowedH5Dataset(
            fold_dir / "val.h5", include_peak_indices=True, **DS_KWARGS
        )
        test_base = MMECGWindowedH5Dataset(
            fold_dir / "test.h5", include_peak_indices=True, **DS_KWARGS
        )
        train_data = train_base
        val_data   = val_base
        test_data  = test_base
        val_wsst_arr  = _load_wsst_h5(fold_dir / "val.h5")
        test_wsst_arr = test_wsst
        train_wsst_full = train_wsst
    else:
        raise ValueError(f"Unknown protocol: {protocol!r}")

    train_ds = RadarODEDataset(train_data, include_peak_indices=False,
                               wsst_cache=train_wsst_full)
    val_ds   = RadarODEDataset(val_data,   include_peak_indices=False,
                               wsst_cache=val_wsst_arr)
    test_ds  = RadarODEDataset(test_data,  include_peak_indices=True,
                               wsst_cache=test_wsst_arr)

    sampler = _subject_sampler(train_data) if balanced_sampling else None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=num_workers, pin_memory=True,
        drop_last=True, collate_fn=collate_radarode,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_radarode,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_radarode,
    )
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────
#  Evaluation helpers
# ─────────────────────────────────────────────────────────────────

def _ppi_targets_from_ridx(r_idx_list: list[np.ndarray],
                           device: torch.device) -> torch.Tensor:
    """Return class labels in [0, 259] from the number of valid RR intervals.

    The original radarODE-MTL PPI target is a padded vector and its CE loss
    effectively supervises the count of non-padding entries. MMECG has direct
    R-peak annotations, so the comparable target is the number of RR intervals
    in the segment.
    """
    labels = []
    for r in r_idx_list:
        n_rr = max(0, int(len(r)) - 1)
        labels.append(min(259, n_rr))
    return torch.tensor(labels, dtype=torch.long, device=device)


def _anchor_targets_from_rpeak(rpeak: torch.Tensor) -> torch.Tensor:
    """Downsample MMECG 1600-point R-peak soft mask to radarODE's 800 anchors."""
    return F.interpolate(rpeak, size=800, mode="linear", align_corners=False)


def radarode_mtl_loss(
    pred: dict[str, torch.Tensor],
    batch: dict,
    device: torch.device,
    lambda_ppi: float,
    lambda_anchor: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ecg = batch["ecg"].to(device)
    rpeak = batch["rpeak"].to(device)
    ppi_target = _ppi_targets_from_ridx(batch["r_idx"], device)
    anchor_target = _anchor_targets_from_rpeak(rpeak)

    shape_loss = F.mse_loss(pred["ECG_shape"], ecg)
    ppi_loss = F.cross_entropy(pred["PPI"].squeeze(1), ppi_target)
    anchor_loss = F.mse_loss(torch.sigmoid(pred["Anchor"]), anchor_target)
    total = shape_loss + lambda_ppi * ppi_loss + lambda_anchor * anchor_loss
    return total, {
        "shape": float(shape_loss.detach().cpu().item()),
        "ppi": float(ppi_loss.detach().cpu().item()),
        "anchor": float(anchor_loss.detach().cpu().item()),
        "total": float(total.detach().cpu().item()),
    }

@torch.no_grad()
def evaluate_val(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    f1_every: int,
) -> dict[str, float]:
    from src.utils.metrics import compute_all_metrics
    model.eval()
    preds, gts, losses = [], [], []
    criterion = nn.MSELoss()
    for batch in loader:
        stft = batch["stft"].to(device)
        ecg  = batch["ecg"].to(device)
        pred = model(stft)
        ecg_pred = pred["ECG_shape"]
        losses.append(float(criterion(ecg_pred, ecg).item()))
        preds.append(ecg_pred.cpu())
        gts.append(batch["ecg"])
    model.train()
    if not preds:
        return {"mae": float("nan"), "rmse": float("nan"),
                "pcc": float("nan"), "loss": float("nan")}
    pred_t = torch.cat(preds, dim=0)
    gt_t   = torch.cat(gts, dim=0)
    metrics = compute_all_metrics(
        pred_t, gt_t, compute_f1=(epoch % f1_every == 0)
    )
    metrics["loss"] = float(np.mean(losses))
    return metrics


# ─────────────────────────────────────────────────────────────────
#  train_fold
# ─────────────────────────────────────────────────────────────────

def train_fold(args: argparse.Namespace, fold: int) -> None:
    random.seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    torch.manual_seed(args.seed + fold)
    torch.cuda.manual_seed_all(args.seed + fold)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir    = Path("experiments_mmecg") / args.exp_tag / f"fold_{fold:02d}"
    ckpt_dir   = run_dir / "checkpoints"
    result_dir = run_dir / "results"
    log_dir    = run_dir / "logs"
    for d in (ckpt_dir, result_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = build_loaders(
        fold=fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        protocol=args.protocol,
        calib_n_train=args.calib_n_train,
        calib_n_val=args.calib_n_val,
        calib_seed=args.calib_seed,
        balanced_sampling=args.balanced_sampling,
    )

    model = RadarODEMTLModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = SGD(model.parameters(), lr=args.lr,
                    momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    cfg_dict = vars(args).copy()
    cfg_dict.update(
        fold=fold,
        num_params=n_params,
        model_name="RadarODE-MTL (adapted)",
        backbone="DCNResNet-50 + SqueezeModule",
        heads="ECG_shape CNNLSTMDecoder + PPI_decoder(260) + anchor_decoder(800)",
        spectrogram_params="WSST: resample 200→30Hz, ssq_cwt Morlet nv=10, [:71,::2] → (50,71,120)",
        output_upsampling="F.interpolate 200→1600",
        loss="MSE(ECG_shape) + lambda_ppi*CE(PPI) + lambda_anchor*MSE(sigmoid(Anchor), R-peak mask)",
    )
    (run_dir / "config.json").write_text(
        json.dumps(cfg_dict, indent=2, default=str)
    )

    best_val_pcc = -float("inf")
    no_improve = 0
    history: list[dict] = []

    with open(log_dir / "train.log", "w") as lf:
        def log(msg: str) -> None:
            print(msg, flush=True)
            lf.write(msg + "\n")
            lf.flush()

        log(f"radarODE-MTL adapted | {args.protocol} fold_{fold:02d}")
        log(f"Train={len(train_loader.dataset)} Val={len(val_loader.dataset)} "
            f"Params={n_params:,}")
        log(f"epochs={args.epochs} patience={args.patience} "
            f"lr={args.lr} mom={args.momentum} wd={args.weight_decay}")
        log(f"MTL weights: lambda_ppi={args.lambda_ppi} "
            f"lambda_anchor={args.lambda_anchor}")

        for epoch in range(1, args.epochs + 1):
            model.train()
            t0 = time.time()
            ep_losses: list[float] = []
            for batch in train_loader:
                stft = batch["stft"].to(device)
                pred = model(stft)
                loss, loss_parts = radarode_mtl_loss(
                    pred, batch, device,
                    lambda_ppi=args.lambda_ppi,
                    lambda_anchor=args.lambda_anchor,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                ep_losses.append(float(loss.item()))
            scheduler.step()
            cur_lr = scheduler.get_last_lr()[0]
            row: dict = {
                "epoch": epoch,
                "train_loss": float(np.mean(ep_losses)),
            }

            if epoch % args.val_every == 0:
                val_m = evaluate_val(
                    model, val_loader, device,
                    epoch=epoch, f1_every=args.f1_every,
                )
                val_f1  = val_m.get("rpeak_f1", float("nan"))
                score_f1 = 0.0 if not np.isfinite(val_f1) else val_f1
                val_score = (val_m.get("pcc", 0.0)
                             - val_m.get("rmse", 0.0)
                             + 0.10 * score_f1)
                row.update({f"val_{k}": v for k, v in val_m.items()})

                ckpt = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "opt": optimizer.state_dict(),
                    "val_pcc": val_m.get("pcc"),
                    "val_rmse": val_m.get("rmse"),
                    "config": cfg_dict,
                }
                pcc_improved = val_m["pcc"] > best_val_pcc
                if pcc_improved:
                    best_val_pcc = val_m["pcc"]
                    no_improve = 0
                    torch.save(ckpt, ckpt_dir / "best.pt")
                else:
                    if epoch >= args.min_epochs:
                        no_improve += args.val_every

                log(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"train={row['train_loss']:.4f} | "
                    f"val_mae={val_m.get('mae', float('nan')):.4f} | "
                    f"val_rmse={val_m.get('rmse', float('nan')):.4f} | "
                    f"val_pcc={val_m.get('pcc', float('nan')):.4f} | "
                    f"val_f1={val_f1:.4f} | "
                    f"val_loss={val_m.get('loss', float('nan')):.4f} | "
                    f"lr={cur_lr:.2e} | "
                    f"{time.time() - t0:.1f}s"
                )
                if (epoch >= args.min_epochs and
                        no_improve >= args.patience):
                    log(f"  -> Early stop at epoch {epoch} "
                        f"(val_pcc no improvement for {args.patience} epochs)")
                    history.append(row)
                    break
            else:
                log(
                    f"Epoch {epoch:03d}/{args.epochs} "
                    f"train_loss={row['train_loss']:.4f} "
                    f"lr={cur_lr:.2e} "
                    f"time={time.time() - t0:.1f}s"
                )
            history.append(row)

    (result_dir / "train_history.json").write_text(
        json.dumps(history, indent=2)
    )


# ─────────────────────────────────────────────────────────────────
#  test_fold
# ─────────────────────────────────────────────────────────────────

def test_fold(args: argparse.Namespace, fold: int) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir    = Path("experiments_mmecg") / args.exp_tag / f"fold_{fold:02d}"
    result_dir = run_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = build_loaders(
        fold=fold,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        protocol=args.protocol,
        calib_n_train=args.calib_n_train,
        calib_n_val=args.calib_n_val,
        calib_seed=args.calib_seed,
        balanced_sampling=False,
    )
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        print(f"  [skip fold {fold}] no checkpoint at {ckpt_path}")
        return pd.DataFrame()

    model = RadarODEMTLModel().to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])

    rows: list[dict] = []
    beat_rows: list[dict] = []
    seg_id = 0
    print(f"\nTesting radarODE fold_{fold:02d}  "
          f"N={len(test_loader.dataset)}  ckpt_epoch={ckpt.get('epoch')}")

    with torch.no_grad():
        for batch in test_loader:
            stft = batch["stft"].to(device)
            pred_t = model(stft)["ECG_shape"].cpu().numpy()[:, 0]  # (B, 1600)
            gt_t   = batch["ecg"].numpy()[:, 0]             # (B, 1600)
            for i in range(pred_t.shape[0]):
                row, beats = _evaluate_segment(
                    seg_id=seg_id,
                    pred_1d=pred_t[i],
                    gt_1d=gt_t[i],
                    gt_r=batch["r_idx"][i],
                    gt_q=batch["q_idx"][i],
                    gt_s=batch["s_idx"][i],
                    gt_t=batch["t_idx"][i],
                    subject_id=int(batch["subject"][i].item()),
                    state_code=int(batch["state"][i].item()),
                    delin_valid=bool(batch["delin_valid"][i].item()),
                )
                rows.append(row)
                beat_rows.extend(beats)
                seg_id += 1
            if seg_id % 200 < args.eval_batch_size:
                print(f"  evaluated {seg_id}/{len(test_loader.dataset)}")

    seg_df  = pd.DataFrame(rows)
    beat_df = pd.DataFrame(beat_rows)
    subj_df = summarize_subject_metrics(rows)
    global_dict = summarize_global_metrics(rows)

    seg_df.to_csv(result_dir / "segment_metrics.csv",  index=False)
    beat_df.to_csv(result_dir / "beat_metrics.csv",    index=False)
    subj_df.to_csv(result_dir / "subject_summary.csv", index=False)
    (result_dir / "global_summary.json").write_text(
        json.dumps(global_dict, indent=2, default=_json_default)
    )
    print(f"  fold_{fold:02d}: pcc={global_dict.get('pcc_raw_mean', 'N/A'):.4f}  "
          f"rmse={global_dict.get('rmse_norm_mean', 'N/A'):.4f}")
    return seg_df


# ─────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="radarODE-MTL adapted for MMECG calib40v10"
    )
    parser.add_argument("--exp_tag", type=str,
                        default="mmecg_radarode_calib40v10")
    parser.add_argument("--protocol", type=str, default="loso_calib",
                        choices=["loso_calib", "loso"])
    parser.add_argument("--fold_idx", type=int, default=1,
                        help="1-based fold index; -1 = all 11 folds")
    parser.add_argument("--mode", type=str, default="train_test",
                        choices=["train_test", "train", "test"])
    # data
    parser.add_argument("--calib_n_train", type=int, default=40)
    parser.add_argument("--calib_n_val",   type=int, default=10)
    parser.add_argument("--calib_seed",    type=int, default=42)
    parser.add_argument("--balanced_sampling", type=lambda x: x.lower() != "false",
                        default=True)
    # training
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=5e-3)
    parser.add_argument("--momentum",   type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip",  type=float, default=0.0)
    parser.add_argument("--lambda_ppi", type=float, default=0.1)
    parser.add_argument("--lambda_anchor", type=float, default=1.0)
    parser.add_argument("--patience",   type=int,   default=50,
                        help="early stop if val_pcc no improvement for N epochs")
    parser.add_argument("--min_epochs", type=int,   default=30)
    parser.add_argument("--val_every",  type=int,   default=5)
    parser.add_argument("--f1_every",   type=int,   default=5)
    parser.add_argument("--num_workers", type=int,  default=2)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    folds = list(range(1, 12)) if args.fold_idx == -1 else [args.fold_idx]

    for fold in folds:
        if args.mode in ("train_test", "train"):
            print(f"\n{'='*60}")
            print(f"Training fold {fold:02d}")
            train_fold(args, fold)
        if args.mode in ("train_test", "test"):
            test_fold(args, fold)

    if args.fold_idx == -1 and args.mode in ("train_test", "test"):
        exp_dir = Path("experiments_mmecg") / args.exp_tag
        loso_dir = exp_dir / "loso_summary"
        loso_dir.mkdir(parents=True, exist_ok=True)
        seg_dfs = []
        for f in range(1, 12):
            seg_csv = exp_dir / f"fold_{f:02d}" / "results" / "segment_metrics.csv"
            if seg_csv.exists():
                seg_dfs.append(pd.read_csv(seg_csv))
        if seg_dfs:
            _loso_summary(seg_dfs, loso_dir)
            summary_path = loso_dir / "global_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                print(f"\n{'='*60}")
                print(f"LOSO summary ({len(fold_dirs)} folds):")
                print(f"  PCC  = {summary.get('pcc_raw_mean', 'N/A'):.4f} ± "
                      f"{summary.get('pcc_raw_std', 'N/A'):.4f}")
                print(f"  RMSE = {summary.get('rmse_norm_mean', 'N/A'):.4f}")
                print(f"  QMR  = {summary.get('qmr_mean', 'N/A'):.1%}")


if __name__ == "__main__":
    main()
