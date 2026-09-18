"""Jin et al. 2025 (IEEE Sensors Journal) -- Attention-DCGAN.

Paper architecture (Sec. III-B, Tables II and III):

  - generator: dense layer, four Conv1DTranspose layers with 512 / 256 / 128 / 64
    kernels of size 5 and stride 1, each followed by batch norm, dropout and
    LeakyReLU, then a self-attention layer on the last deconvolution output, and
    a dense output layer with Tanh;
  - discriminator: dense layer, four Conv1D layers with 512 / 256 / 128 / 64
    kernels of size 5 and stride 1, LeakyReLU after each, then a dense output
    layer with sigmoid;
  - objective: the standard non-saturating GAN game.

See `ADAPTATIONS["attention_dcgan"]` -- notably the paper's R-peak-centred
200-sample fragments are replaced by whole 8 s windows, since ECG-derived
segmentation is a ground-truth oracle under the continuous contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

WIN = 1600


class SelfAttention1d(nn.Module):
    """Scaled dot-product self-attention over time, as written in the paper."""

    def __init__(self, channels: int, key_channels: int | None = None) -> None:
        super().__init__()
        key_channels = key_channels or max(channels // 8, 8)
        self.q = nn.Conv1d(channels, key_channels, 1)
        self.k = nn.Conv1d(channels, key_channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = key_channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x).transpose(1, 2)                       # (B, T, Ck)
        k = self.k(x)                                       # (B, Ck, T)
        v = self.v(x).transpose(1, 2)                       # (B, T, C)
        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2)            # (B, C, T)
        return x + self.gamma * out


class AttentionDCGANGenerator(nn.Module):
    """Four-layer deconvolution generator with one self-attention layer."""

    def __init__(self, in_bins: int = 50, dropout: float = 0.1,
                 negative_slope: float = 0.2, out_gain: float = 1.0) -> None:
        super().__init__()
        self.in_bins = in_bins
        # The paper normalises ECG into the Tanh range. The frozen contract scores a
        # train-fold robust-scaled target instead, so Tanh is followed by a fixed gain
        # fitted on the TRAINING fold only; the evaluator's inverse is unchanged.
        self.register_buffer("out_gain", torch.tensor(float(out_gain)))
        self.stem = nn.Conv1d(in_bins, 512, 1)              # the paper's dense layer
        widths = [(512, 512), (512, 256), (256, 128), (128, 64)]
        blocks = []
        for in_ch, out_ch in widths:
            blocks += [
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.Dropout(dropout),
                nn.LeakyReLU(negative_slope, inplace=True),
            ]
        self.deconv = nn.Sequential(*blocks)
        self.attention = SelfAttention1d(64)
        self.out = nn.Conv1d(64, 1, 1)                      # dense output layer

    def forward(self, wsst: torch.Tensor | None = None, rcg: torch.Tensor | None = None):
        if rcg is None:
            raise ValueError("attention_dcgan consumes the native-rate rcg tensor")
        if rcg.ndim != 3 or rcg.shape[1] != self.in_bins:
            raise ValueError(f"rcg must be [B,{self.in_bins},T], got {tuple(rcg.shape)}")
        h = self.attention(self.deconv(self.stem(rcg)))
        y = self.out_gain * torch.tanh(self.out(h))
        if y.shape[-1] != WIN:
            y = F.interpolate(y, size=WIN, mode="linear", align_corners=False)
        return y, None


class DCGANDiscriminator(nn.Module):
    """Four-layer convolutional discriminator over a 1x1600 ECG-like signal."""

    def __init__(self, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Conv1d(1, 512, 1)
        widths = [(512, 512), (512, 256), (256, 128), (128, 64)]
        blocks = []
        for in_ch, out_ch in widths:
            blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=4, padding=2),
                nn.LeakyReLU(negative_slope, inplace=True),
            ]
        self.conv = nn.Sequential(*blocks)
        self.out = nn.Linear(64, 1)                         # dense output layer (logit)

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        h = self.conv(self.stem(ecg))
        return self.out(h.mean(dim=-1))
