"""Liu et al. 2024 (IEEE Sensors Journal) -- cross-modal Conformer autoencoder.

Paper architecture (Sec. IV):

  - convolutional subsampling: three Conv1d layers with kernel size 3, 16 then
    32 then 32 filters, batch norm after each, followed by a linear dropout
    block;
  - encoder: two Conformer blocks, each with two feed-forward modules around a
    multi-head self-attention module and a convolution module;
  - decoder: two Conformer blocks followed by a linear layer that synthesises
    the ECG;
  - training: waveform MSE plus spectrogram MSE, with a deep-feature loss
    against a pretrained ECG-to-ECG autoencoder's embeddings.

`ECGAutoEncoder` implements the pretraining branch; it is trained per fold on
TRAINING-fold ECG only, so the deep-feature term stays inside the protocol. See
`ADAPTATIONS["cross_modal_conformer"]`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

WIN = 1600


class _FeedForward(nn.Module):
    """Conformer half-step feed-forward module (residual weight 0.5)."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.5 * self.net(x)


class _ConvModule(nn.Module):
    """Conformer convolution module: pointwise GLU, depthwise conv, BN, Swish."""

    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, dim * 2, 1)
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.pw2 = nn.Conv1d(dim, dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).transpose(1, 2)
        h = F.glu(self.pw1(h), dim=1)
        h = F.silu(self.bn(self.dw(h)))
        h = self.drop(self.pw2(h)).transpose(1, 2)
        return x + h


class ConformerBlock(nn.Module):
    """FFN -> multi-head self-attention -> convolution -> FFN -> LayerNorm."""

    def __init__(self, dim: int, n_heads: int = 4, kernel_size: int = 31,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.ff1 = _FeedForward(dim, dropout=dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = _ConvModule(dim, kernel_size, dropout)
        self.ff2 = _FeedForward(dim, dropout=dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff1(x)
        h = self.attn_norm(x)
        x = x + self.attn_drop(self.attn(h, h, h, need_weights=False)[0])
        x = self.conv(x)
        return self.norm(self.ff2(x))


class _ConvSubsampling(nn.Module):
    """Three kernel-3 convolutions (16, 32, 32 filters) with batch norm."""

    def __init__(self, in_ch: int, dim: int, stride: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 16, 3, stride=stride, padding=1), nn.BatchNorm1d(16), nn.SiLU(),
            nn.Conv1d(16, 32, 3, stride=stride, padding=1), nn.BatchNorm1d(32), nn.SiLU(),
            nn.Conv1d(32, 32, 3, stride=stride, padding=1), nn.BatchNorm1d(32), nn.SiLU(),
        )
        self.proj = nn.Linear(32, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).transpose(1, 2)                     # (B, T', 32)
        return self.drop(self.proj(h))                      # (B, T', dim)


class _ConvUpsampling(nn.Module):
    """Transposed-convolution mirror of `_ConvSubsampling` (three stride-2 stages)."""

    def __init__(self, dim: int, stride: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(dim, 32, 4, stride=stride, padding=1), nn.BatchNorm1d(32), nn.SiLU(),
            nn.ConvTranspose1d(32, 32, 4, stride=stride, padding=1), nn.BatchNorm1d(32), nn.SiLU(),
            nn.ConvTranspose1d(32, dim, 4, stride=stride, padding=1), nn.BatchNorm1d(dim), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalConformer(nn.Module):
    """mmWave encoder (conv subsampling + 2 Conformer blocks) and shared ECG decoder."""

    def __init__(self, in_bins: int = 50, dim: int = 64, n_heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.in_bins = in_bins
        self.subsample = _ConvSubsampling(in_bins, dim, dropout=dropout)
        self.encoder = nn.ModuleList([ConformerBlock(dim, n_heads, dropout=dropout)
                                      for _ in range(2)])
        self.decoder = nn.ModuleList([ConformerBlock(dim, n_heads, dropout=dropout)
                                      for _ in range(2)])
        # The convolutional subsampling divides time by 8, so the decoder mirrors it with
        # transposed convolutions. Synthesising the ECG by interpolating the 25 Hz latent
        # instead would band-limit the output below the QRS, handicapping the baseline.
        self.upsample = _ConvUpsampling(dim, stride=2)
        self.out = nn.Linear(dim, 1)

    def embed(self, rcg: torch.Tensor) -> torch.Tensor:
        h = self.subsample(rcg)
        for block in self.encoder:
            h = block(h)
        return h                                            # (B, T', dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        for block in self.decoder:
            z = block(z)
        y = self.upsample(z.transpose(1, 2))                # mirror of the subsampling
        y = self.out(y.transpose(1, 2)).transpose(1, 2)     # (B, 1, T)
        if y.shape[-1] != WIN:
            y = F.interpolate(y, size=WIN, mode="linear", align_corners=False)
        return y

    def forward(self, wsst: torch.Tensor | None = None, rcg: torch.Tensor | None = None,
                return_embedding: bool = False):
        if rcg is None:
            raise ValueError("cross_modal_conformer consumes the native-rate rcg tensor")
        if rcg.ndim != 3 or rcg.shape[1] != self.in_bins:
            raise ValueError(f"rcg must be [B,{self.in_bins},T], got {tuple(rcg.shape)}")
        z = self.embed(rcg)
        y = self.decode(z)
        if return_embedding:
            return y, None, z
        return y, None


class ECGAutoEncoder(nn.Module):
    """Pretrained ECG-to-ECG autoencoder supplying the deep-feature target.

    Trained per fold on TRAINING-fold ECG only. Its encoder embedding has the
    same shape as the mmWave encoder's, so the deep-feature loss is a direct MSE
    between the two.
    """

    def __init__(self, dim: int = 64, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.subsample = _ConvSubsampling(1, dim, dropout=dropout)
        self.encoder = nn.ModuleList([ConformerBlock(dim, n_heads, dropout=dropout)
                                      for _ in range(2)])
        self.decoder = nn.ModuleList([ConformerBlock(dim, n_heads, dropout=dropout)
                                      for _ in range(2)])
        self.upsample = _ConvUpsampling(dim, stride=2)
        self.out = nn.Linear(dim, 1)

    def embed(self, ecg: torch.Tensor) -> torch.Tensor:
        h = self.subsample(ecg)
        for block in self.encoder:
            h = block(h)
        return h

    def forward(self, ecg: torch.Tensor):
        z = self.embed(ecg)
        h = z
        for block in self.decoder:
            h = block(h)
        h = self.upsample(h.transpose(1, 2))
        y = self.out(h.transpose(1, 2)).transpose(1, 2)
        if y.shape[-1] != WIN:
            y = F.interpolate(y, size=WIN, mode="linear", align_corners=False)
        return y, z


def stft_magnitude(x: torch.Tensor, n_fft: int = 256, hop: int = 64) -> torch.Tensor:
    """Spectrogram magnitude used by the published spectrogram loss."""
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=hop, window=window,
                      return_complex=True, center=True)
    return spec.abs()
