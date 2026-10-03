"""Event-only feasibility arm for the registered structured REM-Net Gate 1.

This is intentionally not the final event-conditioned decoder.  It reuses the
incumbent WSST representation and temporal backbone and changes only the event
read-out from two R-centred scales to four explicit Q/R/S/T heatmaps.  A failed
Gate 1 therefore stops the more expensive decoder work.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.continuous.ce_net import LATENT_T, WIN, WSSTStem, _TCN


FIDUCIALS = ("q", "r", "s", "t")


class StructuredEventREM(nn.Module):
    """WSST -> incumbent representation/TCN -> full-resolution Q/R/S/T logits."""

    def __init__(self, in_bins: int = 50, width: int = 128,
                 head_width: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_bins = in_bins
        self.stem = WSSTStem(in_bins=in_bins, width=width, dropout=dropout)
        self.backbone = _TCN(width, [1, 2, 4, 8, 16], dropout=dropout)
        self.head = nn.Sequential(
            nn.Conv1d(width, head_width, 5, padding=2, bias=False),
            nn.BatchNorm1d(head_width),
            nn.GELU(),
            nn.Conv1d(head_width, len(FIDUCIALS), 5, padding=2),
        )

    def forward(self, wsst: torch.Tensor) -> torch.Tensor:
        if wsst.ndim != 4 or wsst.shape[1] != self.in_bins:
            raise ValueError(
                f"wsst must be [B,{self.in_bins},F,T], got {tuple(wsst.shape)}")
        latent = self.stem(wsst)
        if latent.shape[-1] != LATENT_T:
            raise RuntimeError(f"stem emitted {latent.shape[-1]} rather than {LATENT_T}")
        latent = self.backbone(latent)
        full = F.interpolate(latent, size=WIN, mode="linear", align_corners=False)
        logits = self.head(full)
        if logits.shape[1:] != (len(FIDUCIALS), WIN):
            raise RuntimeError(f"unexpected event output {tuple(logits.shape)}")
        return logits
