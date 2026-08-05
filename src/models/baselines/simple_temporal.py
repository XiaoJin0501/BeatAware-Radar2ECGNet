"""Simple continuous temporal baselines for MMECG.

These models intentionally avoid BeatAware-specific event conditioning and
alignment modules. They provide runnable Raw-TCN and KI-TCN baselines for the
Q1 representation-controlled benchmark while keeping the same training and
testing contract as BeatAwareRadar2ECGNet:

    forward(radar) -> (ecg_pred, peak_preds)

`peak_preds` is `None`, so the existing TotalLoss reduces to waveform
reconstruction terms for these baselines.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.fmcw_encoder import FMCWRangeEncoder


class _ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * 3
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=7, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class ContinuousTCNBaseline(nn.Module):
    """Lightweight continuous ECG reconstruction baseline.

    Parameters
    ----------
    n_range_bins:
        Number of input range channels.
    hidden:
        TCN hidden channels.
    dropout:
        Dropout probability.
    ki_channels:
        If set, apply the same KI-style FMCWRangeEncoder before the TCN.
    """

    def __init__(
        self,
        n_range_bins: int = 50,
        hidden: int = 128,
        dropout: float = 0.1,
        ki_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.ki_channels = ki_channels
        in_channels = n_range_bins
        if ki_channels is not None:
            self.ki = FMCWRangeEncoder(
                n_range=n_range_bins,
                out_channels=ki_channels,
                selector="se",
            )
            in_channels = ki_channels

        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.tcn = nn.Sequential(
            _ResidualTCNBlock(hidden, dilation=1, dropout=dropout),
            _ResidualTCNBlock(hidden, dilation=2, dropout=dropout),
            _ResidualTCNBlock(hidden, dilation=4, dropout=dropout),
            _ResidualTCNBlock(hidden, dilation=8, dropout=dropout),
            _ResidualTCNBlock(hidden, dilation=16, dropout=dropout),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if hasattr(self, "ki"):
            x = self.ki(x)
        h = self.input_proj(x)
        h = self.tcn(h)
        ecg = torch.sigmoid(self.decoder(h))
        return ecg, None


class WSSTTCNBaseline(nn.Module):
    """Continuous ECG baseline for cached WSST/STFT-like radar tensors.

    Input shape is expected to be `[B, 50, 71, 120]`: range bins act as input
    channels, while frequency and time form a compact time-frequency map.
    """

    def __init__(
        self,
        in_channels: int = 50,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spec_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden // 2),
            nn.GELU(),
            nn.Conv2d(hidden // 2, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.temporal_proj = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.tcn = nn.Sequential(
            _ResidualTCNBlock(hidden, dilation=1, dropout=dropout),
            _ResidualTCNBlock(hidden, dilation=2, dropout=dropout),
            _ResidualTCNBlock(hidden, dilation=4, dropout=dropout),
            _ResidualTCNBlock(hidden, dilation=8, dropout=dropout),
            _ResidualTCNBlock(hidden, dilation=16, dropout=dropout),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if x.ndim != 4:
            raise ValueError(f"WSSTTCNBaseline expects [B,C,F,T], got {tuple(x.shape)}")
        h = self.spec_encoder(x)
        h = self.freq_pool(h).squeeze(2)  # [B, hidden, T]
        h = F.interpolate(h, size=400, mode="linear", align_corners=False)
        h = self.temporal_proj(h)
        h = self.tcn(h)
        h = F.interpolate(h, size=1600, mode="linear", align_corners=False)
        ecg = torch.sigmoid(self.decoder(h))
        return ecg, None


def _ki_channels_from_arch(arch: str) -> int | None:
    match = re.fullmatch(r"ki(\d+)_tcn", arch)
    if match is None:
        return None
    return int(match.group(1))


def build_simple_baseline(
    arch: str,
    n_range_bins: int = 50,
    hidden: int = 128,
    dropout: float = 0.1,
) -> nn.Module:
    """Build a simple continuous baseline.

    Supported benchmark architectures:

    - `tcn`: raw range-bin TCN
    - `ki3_tcn`, `ki8_tcn`, `ki16_tcn`: KI-compressed TCN variants
    - `stft_tcn`, `wsst_tcn`: cached time-frequency TCN variants
    """
    arch = arch.lower()
    if arch in ("stft_tcn", "wsst_tcn"):
        return WSSTTCNBaseline(
            in_channels=n_range_bins,
            hidden=hidden,
            dropout=dropout,
        )

    if arch == "tcn":
        return ContinuousTCNBaseline(
            n_range_bins=n_range_bins,
            hidden=hidden,
            dropout=dropout,
            ki_channels=None,
        )

    ki_channels = _ki_channels_from_arch(arch)
    if ki_channels is not None:
        return ContinuousTCNBaseline(
            n_range_bins=n_range_bins,
            hidden=hidden,
            dropout=dropout,
            ki_channels=ki_channels,
        )

    raise ValueError(
        f"Unsupported simple baseline architecture {arch!r}. "
        "Runnable Q1 benchmark baselines are: tcn, ki3_tcn, ki8_tcn, "
        "ki16_tcn, stft_tcn, wsst_tcn."
    )
