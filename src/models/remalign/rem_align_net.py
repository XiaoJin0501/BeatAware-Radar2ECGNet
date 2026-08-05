"""Minimal REM-AlignNet variants.

This file starts with the smallest testable model:

    WSST input -> multi-scale encoder -> R/QRS event head
               -> optional event-conditioned latent alignment
               -> event-conditioned TCN decoder -> ECG [B,1,1600]

Latent disentanglement and QRS residual refinement are intentionally left for
later ablations. The first alignment variant is deliberately lightweight and
identity-initialized so it can be tested as a controlled follow-up to REM-Event.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.baselines.simple_temporal import _ResidualTCNBlock


class WSSTEncoder(nn.Module):
    """Lightweight multi-scale encoder for `[B, 50, 71, 120]` WSST tensors."""

    def __init__(self, in_channels: int = 50, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, hidden // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden // 2),
            nn.GELU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, hidden // 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(hidden // 2),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Dropout2d(dropout),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"WSSTEncoder expects [B,C,F,T], got {tuple(x.shape)}")
        h = torch.cat([self.branch3(x), self.branch5(x)], dim=1)
        h = self.fuse(h)
        h = self.freq_pool(h).squeeze(2)
        h = F.interpolate(h, size=400, mode="linear", align_corners=False)
        return self.temporal_proj(h)


class EventStructureHead(nn.Module):
    """Predict R-peak and broad QRS-region masks from temporal latent features."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, hidden // 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
        )
        self.r_head = nn.Conv1d(hidden // 2, 1, kernel_size=1)
        self.qrs_head = nn.Conv1d(hidden // 2, 1, kernel_size=1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = F.interpolate(h, size=1600, mode="linear", align_corners=False)
        e = self.net(e)
        r_mask = torch.sigmoid(self.r_head(e))
        qrs_mask = torch.sigmoid(self.qrs_head(e))
        return r_mask, qrs_mask


class REMEventNet(nn.Module):
    """REM-AlignNet MVP with event-conditioned continuous ECG decoding."""

    def __init__(
        self,
        in_channels: int = 50,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = WSSTEncoder(in_channels=in_channels, hidden=hidden, dropout=dropout)
        self.event_head = EventStructureHead(hidden=hidden)
        self.event_proj = nn.Sequential(
            nn.Conv1d(2, hidden, kernel_size=7, padding=3, bias=False),
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        h = self.encoder(x)
        r_mask, qrs_mask = self.event_head(h)

        event_low = F.interpolate(
            torch.cat([r_mask, qrs_mask], dim=1),
            size=h.shape[-1],
            mode="linear",
            align_corners=False,
        )
        h = h + self.event_proj(event_low)
        h = self.tcn(h)
        h = F.interpolate(h, size=1600, mode="linear", align_corners=False)
        ecg = torch.sigmoid(self.decoder(h))

        zeros = torch.zeros_like(r_mask)
        # Existing TotalLoss supervises peak_preds[0] against the R-peak mask.
        return ecg, (r_mask, qrs_mask, zeros)


class EventConditionedAlignment(nn.Module):
    """Predict a smooth latent delay field and resample temporal features.

    The final convolution is zero-initialized, so the module starts as identity
    alignment and only moves features if training finds a useful delay field.
    """

    def __init__(self, hidden: int = 128, max_delay: float = 8.0) -> None:
        super().__init__()
        self.max_delay = float(max_delay)
        self.delay_head = nn.Sequential(
            nn.Conv1d(hidden + 2, hidden // 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, hidden // 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, 1, kernel_size=1),
        )
        nn.init.zeros_(self.delay_head[-1].weight)
        nn.init.zeros_(self.delay_head[-1].bias)

    def forward(self, h: torch.Tensor, event_low: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, t = h.shape
        delay = torch.tanh(self.delay_head(torch.cat([h, event_low], dim=1))) * self.max_delay
        if t <= 1:
            return h, delay

        base = torch.linspace(-1.0, 1.0, t, device=h.device, dtype=h.dtype).view(1, 1, t)
        offset = 2.0 * delay / float(t - 1)
        x = (base - offset).clamp(-1.0, 1.0)
        y = torch.zeros_like(x)
        grid = torch.stack([x.squeeze(1), y.squeeze(1)], dim=-1).unsqueeze(1)
        aligned = F.grid_sample(
            h.unsqueeze(2),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(2)
        return aligned, delay


class REMAlignNet(nn.Module):
    """REM-AlignNet MVP with event-conditioned latent temporal alignment."""

    def __init__(
        self,
        in_channels: int = 50,
        hidden: int = 128,
        dropout: float = 0.1,
        max_delay: float = 8.0,
    ) -> None:
        super().__init__()
        self.encoder = WSSTEncoder(in_channels=in_channels, hidden=hidden, dropout=dropout)
        self.event_head = EventStructureHead(hidden=hidden)
        self.event_proj = nn.Sequential(
            nn.Conv1d(2, hidden, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.alignment = EventConditionedAlignment(hidden=hidden, max_delay=max_delay)
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        h = self.encoder(x)
        r_mask, qrs_mask = self.event_head(h)
        event_low = F.interpolate(
            torch.cat([r_mask, qrs_mask], dim=1),
            size=h.shape[-1],
            mode="linear",
            align_corners=False,
        )
        h = h + self.event_proj(event_low)
        h, delay = self.alignment(h, event_low)
        h = self.tcn(h)
        h = F.interpolate(h, size=1600, mode="linear", align_corners=False)
        ecg = torch.sigmoid(self.decoder(h))

        # Existing TotalLoss supervises peak_preds[0]. Returning delay in the
        # third slot keeps it inspectable without changing the training loop.
        return ecg, (r_mask, qrs_mask, delay)


class REMAlignMTLNet(nn.Module):
    """REM-AlignNet with an explicit 200-point morphology auxiliary head."""

    def __init__(
        self,
        in_channels: int = 50,
        hidden: int = 128,
        dropout: float = 0.1,
        max_delay: float = 8.0,
    ) -> None:
        super().__init__()
        self.encoder = WSSTEncoder(in_channels=in_channels, hidden=hidden, dropout=dropout)
        self.event_head = EventStructureHead(hidden=hidden)
        self.event_proj = nn.Sequential(
            nn.Conv1d(2, hidden, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.alignment = EventConditionedAlignment(hidden=hidden, max_delay=max_delay)
        self.tcn = nn.Sequential(
            _ResidualTCNBlock(hidden, dilation=1, dropout=dropout),
            _ResidualTCNBlock(hidden, dilation=2, dropout=dropout),
            _ResidualTCNBlock(hidden, dilation=4, dropout=dropout),
            _ResidualTCNBlock(hidden, dilation=8, dropout=dropout),
            _ResidualTCNBlock(hidden, dilation=16, dropout=dropout),
        )
        self.full_decoder = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, 1, kernel_size=1),
        )
        self.shape_decoder = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, hidden // 4, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden // 4),
            nn.GELU(),
            nn.Conv1d(hidden // 4, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        h = self.encoder(x)
        r_mask, qrs_mask = self.event_head(h)
        event_low = F.interpolate(
            torch.cat([r_mask, qrs_mask], dim=1),
            size=h.shape[-1],
            mode="linear",
            align_corners=False,
        )
        h = h + self.event_proj(event_low)
        h, delay = self.alignment(h, event_low)
        h = self.tcn(h)

        full_h = F.interpolate(h, size=1600, mode="linear", align_corners=False)
        ecg = torch.sigmoid(self.full_decoder(full_h))

        shape_h = F.interpolate(h, size=200, mode="linear", align_corners=False)
        shape200 = torch.sigmoid(self.shape_decoder(shape_h))
        ecg.shape200_pred = shape200

        return ecg, (r_mask, qrs_mask, delay)


class REMAlignMTLRefineNet(nn.Module):
    """REM-AlignNet-MTL with morphology-guided residual full-ECG refinement."""

    def __init__(
        self,
        in_channels: int = 50,
        hidden: int = 128,
        dropout: float = 0.1,
        max_delay: float = 8.0,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.encoder = WSSTEncoder(in_channels=in_channels, hidden=hidden, dropout=dropout)
        self.event_head = EventStructureHead(hidden=hidden)
        self.event_proj = nn.Sequential(
            nn.Conv1d(2, hidden, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.alignment = EventConditionedAlignment(hidden=hidden, max_delay=max_delay)
        self.tcn = nn.Sequential(
            _ResidualTCNBlock(hidden, dilation=1, dropout=dropout),
            _ResidualTCNBlock(hidden, dilation=2, dropout=dropout),
            _ResidualTCNBlock(hidden, dilation=4, dropout=dropout),
            _ResidualTCNBlock(hidden, dilation=8, dropout=dropout),
            _ResidualTCNBlock(hidden, dilation=16, dropout=dropout),
        )
        self.full_decoder = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, 1, kernel_size=1),
        )
        self.shape_decoder = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, hidden // 4, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden // 4),
            nn.GELU(),
            nn.Conv1d(hidden // 4, 1, kernel_size=1),
        )
        self.refine_head = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden // 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        h = self.encoder(x)
        r_mask, qrs_mask = self.event_head(h)
        event_low = F.interpolate(
            torch.cat([r_mask, qrs_mask], dim=1),
            size=h.shape[-1],
            mode="linear",
            align_corners=False,
        )
        h = h + self.event_proj(event_low)
        h, delay = self.alignment(h, event_low)
        h = self.tcn(h)

        shape_h = F.interpolate(h, size=200, mode="linear", align_corners=False)
        shape200 = torch.sigmoid(self.shape_decoder(shape_h))

        full_h = F.interpolate(h, size=1600, mode="linear", align_corners=False)
        coarse = torch.sigmoid(self.full_decoder(full_h))
        shape_up = F.interpolate(shape_h, size=1600, mode="linear", align_corners=False)
        residual = self.residual_scale * torch.tanh(
            self.refine_head(torch.cat([full_h, shape_up], dim=1))
        )
        gate = F.max_pool1d(r_mask, kernel_size=81, stride=1, padding=40)
        ecg = (coarse + gate * residual).clamp(0.0, 1.0)
        ecg.shape200_pred = shape200
        ecg.coarse_pred = coarse
        ecg.morphology_residual = residual

        return ecg, (r_mask, qrs_mask, delay)
