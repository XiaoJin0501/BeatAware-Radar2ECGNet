"""Calibration-conditioned REM adapter for full-window continuous ECG.

The frozen radar backbone supplies a continuous prediction and event logits.
A short paired calibration prefix supplies a session token.  The token FiLM-
modulates an event-gated residual corrector; the output remains 1600 samples.
The switches expose morphology-only, event-only, and full interaction controls.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibrationEncoder(nn.Module):
    """Encode paired baseline/reference ECG windows into one session token."""

    def __init__(self, token_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, 9, stride=2, padding=4), nn.GELU(),
            nn.Conv1d(32, 48, 9, stride=2, padding=4), nn.GELU(),
            nn.Conv1d(48, token_dim, 9, stride=2, padding=4), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if prediction.shape != reference.shape or prediction.ndim != 3:
            raise ValueError("calibration prediction/reference must share [N,1,T]")
        derivative = F.pad(reference[..., 1:] - reference[..., :-1], (1, 0))
        token = self.net(torch.cat([prediction, reference, derivative], dim=1)).squeeze(-1)
        return token.mean(dim=0, keepdim=True)  # one token per session


class FiLMResidualBlock(nn.Module):
    def __init__(self, width: int, token_dim: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 7, padding=3 * dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 7, padding=3 * dilation, dilation=dilation)
        self.film = nn.Linear(token_dim, width * 2)

    def forward(self, x: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.conv1(x))
        gamma, beta = self.film(token).chunk(2, dim=1)
        h = h * (1 + gamma[..., None]) + beta[..., None]
        return x + self.conv2(F.gelu(h))


class CalibrationConditionedREM(nn.Module):
    """Continuous residual adapter with explicit E, M, and E×M controls.

    `use_morphology=False` replaces the learned calibration token with zero.
    `use_event=False` removes event gating but keeps parameter count unchanged.
    The full arm enables both, so its advantage over matched controls measures
    interaction rather than a simple added-capacity effect.
    """

    def __init__(self, token_dim: int = 64, width: int = 64,
                 use_event: bool = True, use_morphology: bool = True) -> None:
        super().__init__()
        self.use_event = bool(use_event)
        self.use_morphology = bool(use_morphology)
        self.calibration_encoder = CalibrationEncoder(token_dim)
        self.input_proj = nn.Conv1d(4, width, 9, padding=4)
        self.blocks = nn.ModuleList([FiLMResidualBlock(width, token_dim, d)
                                     for d in (1, 2, 4, 8)])
        self.event_gate = nn.Sequential(nn.Conv1d(2, width, 7, padding=3), nn.Sigmoid())
        self.output = nn.Conv1d(width, 1, 7, padding=3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, query_prediction: torch.Tensor, query_event_logits: torch.Tensor,
                calibration_prediction: torch.Tensor,
                calibration_reference: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if query_prediction.ndim != 3 or query_prediction.shape[1] != 1:
            raise ValueError("query_prediction must be [B,1,T]")
        if query_event_logits.shape[:2] != (query_prediction.shape[0], 2):
            raise ValueError("query_event_logits must be [B,2,T]")
        token = self.calibration_encoder(calibration_prediction, calibration_reference)
        token = token.expand(query_prediction.shape[0], -1)
        if not self.use_morphology:
            token = torch.zeros_like(token)
        derivative = F.pad(query_prediction[..., 1:] - query_prediction[..., :-1], (1, 0))
        events = torch.sigmoid(query_event_logits)
        h = self.input_proj(torch.cat([query_prediction, derivative, events], dim=1))
        for block in self.blocks:
            h = block(h, token)
        gate = self.event_gate(query_event_logits)
        effective_gate = gate if self.use_event else torch.ones_like(gate)
        residual = self.output(h * effective_gate)
        return query_prediction + residual, {
            "calibration_token": token,
            "event_gate": effective_gate,
            "residual": residual,
        }


def interaction_contrast(scores: dict[str, float]) -> float:
    """Factorial E×M interaction: full - event - morphology + neither."""
    required = {"neither", "event_only", "morphology_only", "full"}
    if set(scores) != required:
        raise ValueError(f"scores must contain exactly {sorted(required)}")
    return scores["full"] - scores["event_only"] - scores["morphology_only"] + scores["neither"]
