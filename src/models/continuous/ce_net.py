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

    def __init__(self, in_bins: int = 50, width: int = 128, dropout: float = 0.1,
                 hidden: int | None = None):
        """`hidden` widens the internal channel count only; the OUTPUT stays `width`
        so that everything downstream is bit-identical. Used by B2a-wide."""
        super().__init__()
        hidden = width if hidden is None else hidden
        self.net = nn.Sequential(
            nn.Conv2d(in_bins, hidden, 3, padding=1, bias=False), nn.BatchNorm2d(hidden), nn.GELU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(hidden, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.GELU(),
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


class PhaseResidualFusion(nn.Module):
    """Parameter-matched naive/gated fusion of stable context and signed phase."""

    def __init__(self, width: int = 128, out_ch: int = 192, gated: bool = True):
        super().__init__()
        self.gated = gated
        self.wsst_proj = nn.Conv1d(width, out_ch, 1, bias=False)
        self.rcg_proj = nn.Conv1d(width, out_ch, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv1d(width * 2, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.Sigmoid())
        self.norm = nn.GroupNorm(8, out_ch)

    def forward(self, wsst: torch.Tensor, rcg: torch.Tensor) -> torch.Tensor:
        phase = self.rcg_proj(rcg)
        gate = self.gate(torch.cat([wsst, rcg], 1))
        if not self.gated:
            gate = torch.ones_like(gate)
        return self.norm(self.wsst_proj(wsst) + gate * phase)


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
    """B0/B1 decoder: the incumbent's shape -- interpolate then conv.

    `event_stop_grad` (ablation ladder rung R2 only; default off, so every frozen run is
    unchanged) detaches the backbone features feeding the anchor head: anchor
    supervision then trains the head but sends no gradient into the shared stem/TCN.
    """

    def __init__(self, in_ch=128, width=64, sigmoid=True, event_stop_grad: bool = False):
        super().__init__()
        self.event_stop_grad = event_stop_grad
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
        ev = self.event(h.detach() if self.event_stop_grad else h)
        return (torch.sigmoid(y) if self.sigmoid else y), ev


class CENet(nn.Module):
    """Continuous ECG arms, including matched zero-shot phase-fusion controls."""

    def __init__(self, arm: str, in_bins: int = 50, width: int = 128, dropout: float = 0.1,
                 event_stop_grad: bool = False):
        super().__init__()
        if arm not in ("B0", "B1", "B2a", "B2a-wide", "B2b", "P0-naive", "P1-gated", "R0-rcg"):
            raise ValueError(arm)
        if event_stop_grad and arm not in ("B1", "R0-rcg"):
            raise ValueError("event_stop_grad is defined only for the LegacyDecoder arms")
        self.arm = arm
        self.in_bins = in_bins
        if arm == "R0-rcg":
            # Ablation ladder rung R0: raw range profile through the strided RCG stem,
            # no time-frequency representation; decoder identical to B1. Not a fusion arm.
            self.uses_rcg = True
            self.rcg_stem = RCGStem(in_bins, width)
            self.decoder = LegacyDecoder(width, sigmoid=False, event_stop_grad=event_stop_grad)
            self.fuse = None
            return
        self.uses_rcg = arm in ("B2b", "P0-naive", "P1-gated")
        # B2a-wide: capacity control. WSST stem widened so total params match B2b,
        # while decoder/fuse/output stay identical to B2a. A B2b gain over B2a that
        # does not survive B2a-wide is a capacity effect, not an information effect.
        stem_hidden = 243 if arm == "B2a-wide" else None
        self.wsst_stem = WSSTStem(in_bins, width, dropout, hidden=stem_hidden)
        if arm in ("B0", "B1"):
            self.decoder = LegacyDecoder(width, sigmoid=(arm == "B0"),
                                         event_stop_grad=event_stop_grad)
            self.fuse = None
        elif arm in ("P0-naive", "P1-gated"):
            self.rcg_stem = RCGStem(in_bins, width)
            self.fuse = PhaseResidualFusion(width, 192, gated=(arm == "P1-gated"))
            self.decoder = ContinuousDecoder(192, dropout=dropout)
        else:
            self.rcg_stem = RCGStem(in_bins, width) if self.uses_rcg else None
            fuse_in = width * 2 if self.uses_rcg else width
            self.fuse = nn.Conv1d(fuse_in, 192, 1, bias=False)
            self.decoder = ContinuousDecoder(192, dropout=dropout)

    def forward(self, wsst, rcg=None):
        if self.arm == "R0-rcg":
            if rcg is None or rcg.ndim != 3 or rcg.shape[1:] != (self.in_bins, WIN):
                raise ValueError(f"R0-rcg requires rcg [B,{self.in_bins},{WIN}]")
            # per-window, per-channel standardisation: the RCG analogue of the WSST
            # pipeline's per-window, per-channel normalisation
            rcg = (rcg - rcg.mean(dim=-1, keepdim=True)) / rcg.std(dim=-1, keepdim=True).clamp_min(1e-6)
            y, ev = self.decoder(self.rcg_stem(rcg))
            assert y.shape[-1] == WIN
            return y, ev
        if wsst.ndim != 4 or wsst.shape[1] != self.in_bins:
            raise ValueError(
                f"wsst must have shape [B,{self.in_bins},F,T], got {tuple(wsst.shape)}")
        z = self.wsst_stem(wsst)
        if z.ndim != 3 or z.shape[-1] != LATENT_T:
            raise RuntimeError(f"WSST stem must emit [B,C,{LATENT_T}], got {tuple(z.shape)}")
        if self.fuse is not None:
            if self.uses_rcg:
                if rcg is None:
                    raise ValueError(f"{self.arm} requires the rcg input")
                if rcg.ndim != 3 or rcg.shape[1:] != (self.in_bins, WIN):
                    raise ValueError(
                        f"rcg must have shape [B,{self.in_bins},{WIN}], got {tuple(rcg.shape)}")
                if self.arm in ("P0-naive", "P1-gated"):
                    rcg = (rcg - rcg.mean(dim=-1, keepdim=True)) / rcg.std(dim=-1, keepdim=True).clamp_min(1e-6)
                    z = self.fuse(z, self.rcg_stem(rcg))
                else:
                    rz = self.rcg_stem(rcg)
                    if rz.shape[0] != z.shape[0] or rz.shape[-1] != z.shape[-1]:
                        raise RuntimeError(
                            f"WSST/RCG latent mismatch: {tuple(z.shape)} versus {tuple(rz.shape)}")
                    z = torch.cat([z, rz], dim=1)
                    z = self.fuse(z)
            else:
                z = self.fuse(z)
        y, ev = self.decoder(z)
        assert y.shape[-1] == WIN, f"output {y.shape[-1]} != {WIN}"
        return y, ev


def reset_module_parameters_paired(module: nn.Module, seed: int) -> None:
    """Reset one matched component from an isolated CPU RNG stream.

    Variant-specific stems consume different numbers of random draws. Calling
    this on identically shaped downstream modules makes their initial tensors
    exactly paired without coupling the variant-specific parameters.
    """
    devices = sorted({p.device.index for p in module.parameters()
                      if p.is_cuda and p.device.index is not None})
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        module.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)


def param_report(m: nn.Module) -> dict:
    per = {n: sum(p.numel() for p in mod.parameters() if p.requires_grad)
           for n, mod in m.named_children()}
    return {"total": sum(p.numel() for p in m.parameters() if p.requires_grad), "per_module": per}
