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


def _grid_warp_time(feat: torch.Tensor, offset_steps: torch.Tensor) -> torch.Tensor:
    """Warp `feat` [B,C,L] along time by `offset_steps` [B,1,L] (index units).

    Positive offset delays the signal. Shared by latent-space and output-space
    alignment so the two only differ in where the delay is applied.
    """
    b, c, L = feat.shape
    if L <= 1:
        return feat
    base = torch.linspace(-1.0, 1.0, L, device=feat.device, dtype=feat.dtype).view(1, 1, L)
    offset = 2.0 * offset_steps / float(L - 1)
    x = (base - offset).clamp(-1.0, 1.0)
    y = torch.zeros_like(x)
    grid = torch.stack([x.squeeze(1), y.squeeze(1)], dim=-1).unsqueeze(1)
    return F.grid_sample(
        feat.unsqueeze(2), grid, mode="bilinear", padding_mode="border", align_corners=True
    ).squeeze(2)


class HybridElectromechanicalAlignment(nn.Module):
    """Hybrid electromechanical delay field: interpretable delta_sys + residual.

        delta(t) = delta_sys(t) + r(t)          [milliseconds]

    - `delta_sys(t)`: low-DoF, interpretable systematic delay. A K-knot spline
      whose knot values are predicted from a global latent context, tanh-bounded
      to +/- `delta_sys_max_ms`, zero-initialized so the module starts at identity.
      This is the scientific quantity (approx. PEP) reported/validated for C3.
    - `r(t)`: eps-bounded residual field for local jitter, zero-initialized.

    The delay is defined in milliseconds and applied as a smooth `grid_sample`
    warp of the temporal latent (ms semantics for interpretability, latent warp
    for training stability). `use_delta_sys` and `use_delta_residual` toggle the
    two components independently so one backbone realizes the B0-B4 ablation.
    See docs/ALIGNMENT_DESIGN.md.
    """

    def __init__(
        self,
        hidden: int = 128,
        window_ms: float = 8000.0,
        knots: int = 5,
        delta_sys_max_ms: float = 150.0,
        delta_residual_eps_ms: float = 30.0,
        use_delta_sys: bool = True,
        use_delta_residual: bool = True,
        warp_latent: bool = True,
    ) -> None:
        super().__init__()
        self.window_ms = float(window_ms)
        self.knots = int(knots)
        self.delta_sys_max_ms = float(delta_sys_max_ms)
        self.delta_residual_eps_ms = float(delta_residual_eps_ms)
        self.use_delta_sys = bool(use_delta_sys)
        self.use_delta_residual = bool(use_delta_residual)
        # When False, the module still predicts delta but leaves the latent
        # untouched, so the model can apply delta at output/event time instead
        # (align_space="output"). This tests the "latent warp is absorbed by the
        # decoder" hypothesis. See docs/ALIGNMENT_DESIGN.md.
        self.warp_latent = bool(warp_latent)

        # Systematic delay: K knot values from a global latent context.
        self.knot_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, self.knots),
        )
        nn.init.zeros_(self.knot_head[-1].weight)
        nn.init.zeros_(self.knot_head[-1].bias)

        # Residual jitter field conditioned on latent + event map.
        self.residual_head = nn.Sequential(
            nn.Conv1d(hidden + 2, hidden // 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, 1, kernel_size=1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self, h: torch.Tensor, event_low: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, t = h.shape

        if self.use_delta_sys:
            ctx = h.mean(dim=-1)  # [B, C] global context
            knots = torch.tanh(self.knot_head(ctx)) * self.delta_sys_max_ms  # [B, K]
            delta_sys = F.interpolate(
                knots.unsqueeze(1), size=t, mode="linear", align_corners=True
            )  # [B, 1, t] ms
        else:
            delta_sys = torch.zeros(b, 1, t, device=h.device, dtype=h.dtype)

        if self.use_delta_residual:
            delta_res = self.delta_residual_eps_ms * torch.tanh(
                self.residual_head(torch.cat([h, event_low], dim=1))
            )  # [B, 1, t] ms
        else:
            delta_res = torch.zeros(b, 1, t, device=h.device, dtype=h.dtype)

        delta_ms = delta_sys + delta_res
        if (not self.warp_latent) or t <= 1:
            return h, delta_sys, delta_res

        # ms -> latent-step offset: the latent spans `t` steps over `window_ms`.
        ms_per_step = self.window_ms / float(t)
        delta_steps = delta_ms / ms_per_step
        aligned = _grid_warp_time(h, delta_steps)
        return aligned, delta_sys, delta_res


class REMAlignHybridMTLNet(nn.Module):
    """REM-AlignNet-MTL with the hybrid electromechanical alignment module.

    Identical to REMAlignMTLNet except the latent alignment is the ablatable
    HybridElectromechanicalAlignment (delta_sys + residual). The 200-point
    morphology head is retained so Shape-200 stays comparable across the B0-B4
    ablation. delta_sys / delta_residual are exposed on the output tensor for
    interpretability analysis (C3).
    """

    def __init__(
        self,
        in_channels: int = 50,
        hidden: int = 128,
        dropout: float = 0.1,
        knots: int = 5,
        delta_sys_max_ms: float = 150.0,
        delta_residual_eps_ms: float = 30.0,
        use_delta_sys: bool = True,
        use_delta_residual: bool = True,
        align_space: str = "latent",
        fs: int = 200,
        out_len: int = 1600,
    ) -> None:
        super().__init__()
        self.align_space = str(align_space).lower()
        self.fs = int(fs)
        self.out_len = int(out_len)
        self.encoder = WSSTEncoder(in_channels=in_channels, hidden=hidden, dropout=dropout)
        self.event_head = EventStructureHead(hidden=hidden)
        self.event_proj = nn.Sequential(
            nn.Conv1d(2, hidden, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.alignment = HybridElectromechanicalAlignment(
            hidden=hidden,
            knots=knots,
            delta_sys_max_ms=delta_sys_max_ms,
            delta_residual_eps_ms=delta_residual_eps_ms,
            use_delta_sys=use_delta_sys,
            use_delta_residual=use_delta_residual,
            warp_latent=(self.align_space == "latent"),
        )
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
        h, delta_sys, delta_res = self.alignment(h, event_low)
        h = self.tcn(h)

        full_h = F.interpolate(h, size=self.out_len, mode="linear", align_corners=False)
        ecg = torch.sigmoid(self.full_decoder(full_h))

        if self.align_space == "output":
            # Apply the predicted delay (ms) at output/event time instead of in
            # the latent, so it cannot be absorbed by the decoder.
            delta_out_ms = F.interpolate(
                delta_sys + delta_res, size=self.out_len, mode="linear", align_corners=True
            )
            delta_samples = delta_out_ms * (self.fs / 1000.0)
            ecg = _grid_warp_time(ecg, delta_samples)

        shape_h = F.interpolate(h, size=200, mode="linear", align_corners=False)
        shape200 = torch.sigmoid(self.shape_decoder(shape_h))
        ecg.shape200_pred = shape200
        ecg.delta_sys = delta_sys
        ecg.delta_residual = delta_res

        return ecg, (r_mask, qrs_mask, delta_sys)
