"""Zhang et al. 2025 (IEEE TIM) -- radarODE-MTL, adapted to the continuous contract.

The feature extractor is imported verbatim from the authors' public repository
(`~/Projects/radarODE-MTL`): `DCNResNet` (deformable-convolution ResNet) followed
by `SqueezeModule`, exactly the pair used by `radarODE_MTL.forward`. Only the
output stage is adapted: the published ECG_shape(200) / PPI(260) / anchor(800)
heads cannot emit a continuous 1600-sample stream without the long-term assembly
that this project could not faithfully reproduce (claim ledger F2), so the
continuous decoder below is assembled from the authors' own DeconvBlock /
ConvBlock primitives and labelled `adapted (continuous head)`.

See `ADAPTATIONS["radarode_mtl"]`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

WIN = 1600
RADARODE_ROOT = Path("/home/qhh2237/Projects/radarODE-MTL")


def _import_official_backbone():
    """Import DCNResNet + SqueezeModule from the authors' checkout."""
    project = RADARODE_ROOT / "Projects" / "radarODE_plus"
    if not project.exists():
        raise FileNotFoundError(
            f"radarODE-MTL source not found at {RADARODE_ROOT}. The baseline requires the "
            "authors' repository; do not substitute a re-implementation.")
    for p in (str(RADARODE_ROOT), str(project)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from nets.backbone.dcnresnet_backbone import DCNResNet  # noqa: E402
    from nets.backbone.squeeze_module import SqueezeModule  # noqa: E402
    return DCNResNet, SqueezeModule


class _DeconvBlock1d(nn.Module):
    """The authors' DeconvBlock (ConvTranspose1d + BatchNorm1d + ReLU)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int,
                 padding: int) -> None:
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=kernel_size,
                                                 stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.batch_norm(self.conv_transpose(x)))


class _ConvBlock1d(nn.Module):
    """The authors' ConvBlock (Conv1d + BatchNorm1d + ReLU)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int,
                 padding: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                              padding=padding)
        self.batch_norm = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.batch_norm(self.conv(x)))


class RadarODEMTLContinuous(nn.Module):
    """Official radarODE-MTL backbone with an adapted continuous 1x1600 decoder."""

    def __init__(self, in_bins: int = 50, dropout: float = 0.1) -> None:
        super().__init__()
        DCNResNet, SqueezeModule = _import_official_backbone()
        self.in_bins = in_bins
        self.backbone = DCNResNet(in_bins)
        self.squeeze_module = SqueezeModule(in_channels=1024, out_channels=1024)
        self.decoder = nn.Sequential(
            _DeconvBlock1d(1024, 512, kernel_size=5, stride=3, padding=1),
            _DeconvBlock1d(512, 256, kernel_size=5, stride=3, padding=1),
            _ConvBlock1d(256, 128, kernel_size=7, stride=1, padding=3),
            nn.Dropout(dropout),
            _ConvBlock1d(128, 64, kernel_size=7, stride=1, padding=3),
            _ConvBlock1d(64, 32, kernel_size=5, stride=1, padding=2),
        )
        self.head = nn.Conv1d(32, 1, kernel_size=5, padding=2)

    def forward(self, wsst: torch.Tensor | None = None, rcg: torch.Tensor | None = None):
        if wsst is None:
            raise ValueError("radarode_mtl consumes the WSST tensor")
        if wsst.ndim != 4 or wsst.shape[1] != self.in_bins:
            raise ValueError(f"wsst must be [B,{self.in_bins},F,T], got {tuple(wsst.shape)}")
        h = self.squeeze_module(self.backbone(wsst))                  # (B, 1024, T')
        h = self.decoder(h)
        if h.shape[-1] != WIN:
            h = F.interpolate(h, size=WIN, mode="linear", align_corners=False)
        return self.head(h), None
