"""Chen et al. 2024 (IEEE TMC) -- the MMECG dataset paper's own reconstruction model.

Paper architecture (Sec. 3.3 and 4.2.3):

  - temporal encoder: 4 repeated layers, each two padded 1D convolutions with
    kernel 7, ReLU + batch norm, followed by max-pooling with stride 2;
    applied independently to every one of the 50 spatial signals;
  - spatial encoder: 3 transformer blocks over the 50 signals, 4 attention
    heads, Q/K/V dimension 64, feed-forward dimension 128, with a learned
    embedding of each signal's position;
  - bridge: transposed-convolution upsampling back to the input length, then a
    dot product between the expanded temporal and spatial feature maps;
  - decoder: 9 stacked TCN blocks with dilation factor 2 (receptive field 512),
    causal so a sample never depends on future timesteps.

Adaptations for the continuous contract are listed in `ADAPTATIONS["mmecg_tcn"]`
(range-bin input, parallel continuous head instead of autoregressive mu-law
softmax decoding, bin-index position embedding).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

WIN = 1600


class _TemporalEncoderLayer(nn.Module):
    """Two padded k=7 convolutions (ReLU + BN each) then stride-2 max pooling."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_ch),
            nn.Conv1d(out_ch, out_ch, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_ch),
        )
        self.pool = nn.MaxPool1d(2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.net(x))


class _CausalTCNBlock(nn.Module):
    """Dilated causal residual block (left padding only, then chomp)."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 7,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def _causal(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        return conv(F.pad(x, (self.pad, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.relu(self.norm1(self._causal(self.conv1, x))))
        h = F.relu(self.norm2(self._causal(self.conv2, h)))
        return F.relu(x + h)


class MMECGBaseline(nn.Module):
    """CNN-Transformer spatio-temporal encoder with a causal dilated TCN decoder."""

    def __init__(self, in_bins: int = 50, temporal_ch: int = 32, n_transformer: int = 3,
                 n_heads: int = 4, ff_dim: int = 128, n_tcn: int = 9,
                 tcn_ch: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_bins = in_bins
        self.temporal_ch = temporal_ch

        # Temporal encoder: shared across bins, 4 layers -> length / 16.
        widths = [1, temporal_ch // 4, temporal_ch // 2, temporal_ch, temporal_ch]
        self.temporal = nn.Sequential(*[
            _TemporalEncoderLayer(widths[i], widths[i + 1]) for i in range(4)
        ])

        # Spatial encoder over the 50 bins: pooled temporal descriptor per bin.
        self.bin_embed = nn.Embedding(in_bins, temporal_ch)
        block = nn.TransformerEncoderLayer(
            d_model=temporal_ch, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=False)
        self.spatial = nn.TransformerEncoder(block, num_layers=n_transformer)

        # Bridge: transposed-convolution upsampling of the temporal features.
        self.up = nn.Sequential(
            nn.ConvTranspose1d(temporal_ch, temporal_ch, 4, stride=2, padding=1),
            nn.ReLU(inplace=True), nn.BatchNorm1d(temporal_ch),
            nn.ConvTranspose1d(temporal_ch, temporal_ch // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True), nn.BatchNorm1d(temporal_ch // 2),
            nn.ConvTranspose1d(temporal_ch // 2, temporal_ch // 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True), nn.BatchNorm1d(temporal_ch // 4),
            nn.ConvTranspose1d(temporal_ch // 4, temporal_ch // 8, 4, stride=2, padding=1),
            nn.ReLU(inplace=True), nn.BatchNorm1d(temporal_ch // 8),
        )
        fused_ch = in_bins * (temporal_ch // 8)
        self.fuse = nn.Conv1d(fused_ch, tcn_ch, 1)

        # Decoder: 9 causal dilated TCN blocks, dilation factor 2.
        self.tcn = nn.ModuleList([
            _CausalTCNBlock(tcn_ch, dilation=2 ** i, dropout=dropout) for i in range(n_tcn)
        ])
        self.head = nn.Conv1d(tcn_ch, 1, 1)

    def forward(self, wsst: torch.Tensor | None = None, rcg: torch.Tensor | None = None):
        if rcg is None:
            raise ValueError("mmecg_tcn consumes the native-rate rcg tensor")
        if rcg.ndim != 3 or rcg.shape[1] != self.in_bins:
            raise ValueError(f"rcg must be [B,{self.in_bins},T], got {tuple(rcg.shape)}")
        b, r, t = rcg.shape

        # (B, R, T) -> (B*R, 1, T) so every bin passes the shared temporal encoder.
        h = self.temporal(rcg.reshape(b * r, 1, t))                   # (B*R, C, T/16)
        c, tt = h.shape[1], h.shape[2]

        # Spatial attention over bins, on the time-pooled descriptor of each bin.
        s = h.mean(dim=2).reshape(b, r, c)                            # (B, R, C)
        s = s + self.bin_embed.weight.unsqueeze(0)
        s = self.spatial(s)                                           # (B, R, C)

        # Bridge: upsample the temporal map, then modulate it by the spatial map.
        u = self.up(h)                                                # (B*R, C/8, T)
        u = u.reshape(b, r, -1, u.shape[-1])                          # (B, R, C/8, T)
        gate = s.mean(dim=2).reshape(b, r, 1, 1)                      # (B, R, 1, 1)
        u = (u * gate).reshape(b, -1, u.shape[-1])                    # (B, R*C/8, T)
        if u.shape[-1] != WIN:
            u = F.interpolate(u, size=WIN, mode="linear", align_corners=False)

        z = self.fuse(u)
        for block in self.tcn:
            z = block(z)
        return self.head(z), None
