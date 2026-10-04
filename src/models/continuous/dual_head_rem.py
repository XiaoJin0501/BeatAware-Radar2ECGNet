"""Compact dual-head REM-Net for the registered shape/anchor interaction gate.

The representation is shared.  The shape branch emits one phase-normalised
200-point cardiac cycle; the anchor branch is either the incumbent two-scale
R read-out or the candidate Q/R/S/T read-out.  No continuous waveform decoder
or reference ECG is consumed by ``forward``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.continuous.ce_net import LATENT_T, WSSTStem, _TCN


class DualHeadREM(nn.Module):
    """WSST -> shared stem/TCN -> ECG shape and anchor logits."""

    def __init__(self, anchor_mode: str = "qrst", in_bins: int = 50,
                 width: int = 128, head_width: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if anchor_mode not in {"r_multiscale", "qrst"}:
            raise ValueError(f"unsupported anchor_mode={anchor_mode}")
        self.anchor_mode = anchor_mode
        self.in_bins = in_bins
        self.stem = WSSTStem(in_bins=in_bins, width=width, dropout=dropout)
        self.backbone = _TCN(width, [1, 2, 4, 8, 16], dropout=dropout)
        self.shape_head = nn.Sequential(
            nn.Conv1d(width, head_width, 5, padding=2, bias=False),
            nn.BatchNorm1d(head_width),
            nn.GELU(),
            nn.Conv1d(head_width, 1, 5, padding=2),
        )
        # Keep A0/A1 exactly capacity-matched.  The control consumes only the
        # first two logits; its two unused logits deliberately receive no loss.
        channels = 4
        self.anchor_head = nn.Sequential(
            nn.Conv1d(width, head_width, 5, padding=2, bias=False),
            nn.BatchNorm1d(head_width),
            nn.GELU(),
            nn.Conv1d(head_width, channels, 5, padding=2),
        )

    def forward(self, wsst: torch.Tensor) -> dict[str, torch.Tensor]:
        if wsst.ndim != 4 or wsst.shape[1] != self.in_bins:
            raise ValueError(
                f"wsst must be [B,{self.in_bins},F,T], got {tuple(wsst.shape)}")
        latent = self.backbone(self.stem(wsst))
        if latent.shape[-1] != LATENT_T:
            raise RuntimeError(f"backbone emitted {latent.shape[-1]} rather than {LATENT_T}")
        shape = torch.sigmoid(self.shape_head(
            F.interpolate(latent, size=200, mode="linear", align_corners=False)))
        anchor = self.anchor_head(
            F.interpolate(latent, size=1600, mode="linear", align_corners=False))
        if self.anchor_mode == "r_multiscale":
            anchor = anchor[:, :2]
        expected = 2 if self.anchor_mode == "r_multiscale" else 4
        if shape.shape[1:] != (1, 200) or anchor.shape[1:] != (expected, 1600):
            raise RuntimeError(
                f"unexpected outputs shape={tuple(shape.shape)}, anchor={tuple(anchor.shape)}")
        return {"shape": shape, "anchor_logits": anchor}
