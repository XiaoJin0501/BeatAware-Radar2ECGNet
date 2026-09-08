"""Continuous-ECG pilot arms B0 / B1 / B2a / B2b.

Governed by docs/CONTINUOUS_ECG_TASK_CONTRACT.md and the spec
experiments/specs/q1_continuous_ecg_b0_b2_pilot.json.

Parity rule (spec arm_parity): B2a and B2b share the continuous decoder, fusion,
upsampling stack and output head exactly. The ONLY difference is B2b's native-rate
rcg stem and the widened fusion projection that accepts it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

WIN = 1600
LATENT_T = 400


class WSSTStem(nn.Module):
    """(B, R, 71, T) -> (B, C, LATENT_T). Frequency is pooled; time is preserved."""

    def __init__(self, in_bins: int = 50, width: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_bins, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.GELU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        h = self.freq_pool(self.net(x)).squeeze(2)             # (B, C, T)
        return F.interpolate(h, size=LATENT_T, mode="linear", align_corners=False)


class RCGStem(nn.Module):
    """(B, R, 1600) -> (B, C, LATENT_T) by STRIDED conv, never interpolation.

    Present only in B2b. This is the native-rate path whose contribution the
    B2a-to-B2b contrast is designed to measure.
    """

    def __init__(self, in_bins: int = 50, width: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_bins, width, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(width), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)                                     # 1600 -> 400


class _TCN(nn.Module):
    def __init__(self, c, dils, k=7, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for d in dils:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(c, c, k, padding=d * (k - 1) // 2, dilation=d, bias=False),
                nn.BatchNorm1d(c), nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(c, c, k, padding=d * (k - 1) // 2, dilation=d, bias=False),
                nn.BatchNorm1d(c), nn.GELU()))

    def forward(self, x):
        for b in self.blocks:
            x = x + b(x)
        return x


class ContinuousDecoder(nn.Module):
    """Dual-scale decoder. IDENTICAL in B2a and B2b -- parity requirement."""

    def __init__(self, in_ch=192, width=96, dropout=0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, width * 2, 1, bias=False)
        self.rhythm = _TCN(width, [1, 2, 4, 8, 16], dropout=dropout)     # long context
        self.morph = _TCN(width, [1, 2], k=7, dropout=dropout)           # local detail
        self.gate = nn.Sequential(nn.Conv1d(width * 2, width * 2, 1), nn.Sigmoid())
        self.up = nn.Sequential(
            nn.ConvTranspose1d(width * 2, width, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(width), nn.GELU(),
            nn.ConvTranspose1d(width, width // 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(width // 2), nn.GELU())
        self.out = nn.Conv1d(width // 2, 1, 5, padding=2)                # LINEAR, no sigmoid
        self.event = nn.Conv1d(width // 2, 2, 5, padding=2)

    def forward(self, z):
        h = self.proj(z)
        c = h.shape[1] // 2
        r, m = self.rhythm(h[:, :c]), self.morph(h[:, c:])
        h = torch.cat([r, m], 1)
        h = h * self.gate(h)
        h = self.up(h)
        return self.out(h), self.event(h)


class LegacyDecoder(nn.Module):
    """B0/B1 decoder: the incumbent's shape -- interpolate then conv."""

    def __init__(self, in_ch=128, width=64, sigmoid=True):
        super().__init__()
        self.tcn = _TCN(in_ch, [1, 2, 4, 8, 16])
        self.head = nn.Sequential(nn.Conv1d(in_ch, width, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(width, 1, 5, padding=2))
        self.event = nn.Sequential(nn.Conv1d(in_ch, width, 5, padding=2), nn.GELU(),
                                   nn.Conv1d(width, 2, 5, padding=2))
        self.sigmoid = sigmoid

    def forward(self, z):
        h = self.tcn(z)
        h = F.interpolate(h, size=WIN, mode="linear", align_corners=False)
        y = self.head(h)
        return (torch.sigmoid(y) if self.sigmoid else y), self.event(h)


class CENet(nn.Module):
    """arm in {B0, B1, B2a, B2b}."""

    def __init__(self, arm: str, in_bins: int = 50, width: int = 128, dropout: float = 0.1):
        super().__init__()
        if arm not in ("B0", "B1", "B2a", "B2b"):
            raise ValueError(arm)
        self.arm = arm
        self.uses_rcg = arm == "B2b"
        self.wsst_stem = WSSTStem(in_bins, width, dropout)
        if arm in ("B0", "B1"):
            self.decoder = LegacyDecoder(width, sigmoid=(arm == "B0"))
            self.fuse = None
        else:
            self.rcg_stem = RCGStem(in_bins, width) if self.uses_rcg else None
            fuse_in = width * 2 if self.uses_rcg else width
            self.fuse = nn.Conv1d(fuse_in, 192, 1, bias=False)
            self.decoder = ContinuousDecoder(192, dropout=dropout)

    def forward(self, wsst, rcg=None):
        z = self.wsst_stem(wsst)
        if self.fuse is not None:
            if self.uses_rcg:
                if rcg is None:
                    raise ValueError("B2b requires the rcg input")
                z = torch.cat([z, self.rcg_stem(rcg)], dim=1)
            z = self.fuse(z)
        y, ev = self.decoder(z)
        assert y.shape[-1] == WIN, f"output {y.shape[-1]} != {WIN}"
        return y, ev


def param_report(m: nn.Module) -> dict:
    per = {n: sum(p.numel() for p in mod.parameters() if p.requires_grad)
           for n, mod in m.named_children()}
    return {"total": sum(p.numel() for p in m.parameters() if p.requires_grad), "per_module": per}
