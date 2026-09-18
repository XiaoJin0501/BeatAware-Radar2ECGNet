"""Xu et al. 2025 (Signal Processing) -- WaveGRU-Net, adapted to the continuous contract.

Paper architecture (Sec. 3.2):

  - MODWT front end: sym4 basis, 5 decomposition levels (6 bands), shift
    invariant, applied to the radar phase signal;
  - three convolutional blocks, each Conv1d + batch norm + tanh + max pooling,
    with the kernel size shrinking across layers: 12 kernels of size 64, then
    8 of size 64, then 8 of size 16;
  - a three-layer transposed-convolution + tanh reconstruction module;
  - a bidirectional GRU that merges the forward and flipped paths by
    concatenation, followed by fully connected layers with dropout.

The MODWT is implemented here as a fixed a-trous (undecimated) sym4 filter bank
with non-trainable buffers, so the transform runs on the GPU inside the model
instead of costing CPU time in the data loader. `tests/` checks it against
`pywt.swt` on the same input. See `ADAPTATIONS["wavegru_net"]`.
"""

from __future__ import annotations

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

WIN = 1600
WAVELET = "sym4"
LEVELS = 5


class MODWTLayer(nn.Module):
    """Shift-invariant sym4 decomposition into `levels + 1` bands per channel.

    Bands are ordered detail 1..L then the final approximation, matching the
    paper's 6-band layout at `levels = 5`. Filters are `1/sqrt(2)`-normalised
    per level and applied with circular padding, the MODWT convention.
    """

    def __init__(self, channels: int, wavelet: str = WAVELET, levels: int = LEVELS) -> None:
        super().__init__()
        self.channels, self.levels = channels, levels
        w = pywt.Wavelet(wavelet)
        lo = (np.asarray(w.dec_lo, dtype=np.float64) / np.sqrt(2.0)).astype(np.float32)
        hi = (np.asarray(w.dec_hi, dtype=np.float64) / np.sqrt(2.0)).astype(np.float32)
        # torch conv1d is cross-correlation; flip so this is a true convolution.
        self.register_buffer("lo", torch.from_numpy(lo[::-1].copy()))
        self.register_buffer("hi", torch.from_numpy(hi[::-1].copy()))

    def _filter(self, x: torch.Tensor, taps: torch.Tensor, dilation: int) -> torch.Tensor:
        c = x.shape[1]
        k = taps.numel()
        weight = taps.view(1, 1, k).expand(c, 1, k)
        pad = (k - 1) * dilation
        xp = F.pad(x, (pad, 0), mode="circular")
        return F.conv1d(xp, weight, dilation=dilation, groups=c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, C, T)` -> `(B, C * (levels + 1), T)`."""
        approx, bands = x, []
        for j in range(self.levels):
            dilation = 2 ** j
            bands.append(self._filter(approx, self.hi, dilation))
            approx = self._filter(approx, self.lo, dilation)
        bands.append(approx)
        return torch.cat(bands, dim=1)


class _ConvBlock(nn.Module):
    """Conv1d + BatchNorm + tanh + max pooling, as published."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, pool: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.Tanh(),
            nn.MaxPool1d(pool, stride=pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WaveGRUNet(nn.Module):
    """MODWT front end + 3-block CNN + transposed-conv module + Bi-GRU head."""

    def __init__(self, in_bins: int = 50, gru_hidden: int = 64, dropout: float = 0.1,
                 levels: int = LEVELS, bottleneck: int = 32) -> None:
        super().__init__()
        self.in_bins, self.levels = in_bins, levels
        self.modwt = MODWTLayer(in_bins, levels=levels)
        # Band mixing keeps the published CNN widths affordable with 50 bins x 6 bands.
        self.mix = nn.Sequential(
            nn.Conv1d(in_bins * (levels + 1), bottleneck, 1, bias=False),
            nn.BatchNorm1d(bottleneck), nn.Tanh(),
        )
        self.cnn = nn.Sequential(
            _ConvBlock(bottleneck, 96, kernel_size=64),     # large kernels: low-frequency
            _ConvBlock(96, 64, kernel_size=64),             # QRS transitions
            _ConvBlock(64, 32, kernel_size=16),             # residual noise removal
        )
        self.trans = nn.Sequential(
            nn.ConvTranspose1d(32, 64, 4, stride=2, padding=1), nn.Tanh(),
            nn.ConvTranspose1d(64, 48, 4, stride=2, padding=1), nn.Tanh(),
            nn.ConvTranspose1d(48, 32, 4, stride=2, padding=1), nn.Tanh(),
        )
        self.gru = nn.GRU(32, gru_hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2 * gru_hidden, 64), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, wsst: torch.Tensor | None = None, rcg: torch.Tensor | None = None):
        if rcg is None:
            raise ValueError("wavegru_net consumes the native-rate rcg tensor")
        if rcg.ndim != 3 or rcg.shape[1] != self.in_bins:
            raise ValueError(f"rcg must be [B,{self.in_bins},T], got {tuple(rcg.shape)}")
        h = self.mix(self.modwt(rcg))
        h = self.trans(self.cnn(h))
        h, _ = self.gru(h.transpose(1, 2))                   # (B, T, 2H)
        y = self.fc(h).transpose(1, 2)                       # (B, 1, T)
        if y.shape[-1] != WIN:
            y = F.interpolate(y, size=WIN, mode="linear", align_corners=False)
        return y, None
