"""Zhang et al. 2025 (IEEE TIM) -- radarODE-MTL, adapted to the continuous contract.

Everything that carries the paper's mechanism is imported from the authors'
public repository (`~/Projects/radarODE-MTL`) rather than re-implemented:

  - `DCNResNet` + `SqueezeModule`: the shared deformable-convolution backbone;
  - `shapeDecoder`: the LSTM-CNN encoder/decoder producing the beat shape;
  - `PPI_decoder`: the peak-to-peak-interval head, which is where the ODE lives
    (`ECGParameterEstimator` + `ode1_solver`) -- the "ODE" in radarODE;
  - `anchor_decoder`: the R-peak anchor head;
  - `LibMTL.weighting.EGA`: eccentric gradient alignment, the paper's multitask
    contribution, with the authors' own settings (`EGA_temp = 1`,
    `rep_grad = False`).

Only one head is added: a continuous 1x1600 decoder built from the authors' own
DeconvBlock / ConvBlock primitives, because the published heads are beat-level
and reach a continuous stream only through the long-term assembly this project
could not faithfully reproduce (claim ledger F2). The continuous head is the one
scored; the three published tasks train jointly under EGA and shape the shared
representation exactly as the paper intends.

See `ADAPTATIONS["radarode_mtl"]`.
"""

from __future__ import annotations

import sys
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WIN = 1600
PPI_FRAMES = 31            # the PPI head's linear layer is sized for this rep length
PPI_CLASSES = 260
SHAPE_LEN = 200
RADARODE_ROOT = Path("/home/qhh2237/Projects/radarODE-MTL")
TASKS = ("continuous", "ECG_shape", "PPI", "Anchor")


def _import_official():
    """Import the authors' backbone, decoders and EGA weighting."""
    project = RADARODE_ROOT / "Projects" / "radarODE_plus"
    if not project.exists():
        raise FileNotFoundError(
            f"radarODE-MTL source not found at {RADARODE_ROOT}. The baseline requires the "
            "authors' repository; do not substitute a re-implementation.")
    for p in (str(RADARODE_ROOT), str(project)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from LibMTL.weighting.EGA import EGA  # noqa: E402
    from nets.anchor_decoder import anchor_decoder  # noqa: E402
    from nets.backbone.dcnresnet_backbone import DCNResNet  # noqa: E402
    from nets.backbone.squeeze_module import SqueezeModule  # noqa: E402
    from nets.model import shapeDecoder  # noqa: E402
    from nets.PPI_decoder import PPI_decoder  # noqa: E402
    return EGA, DCNResNet, SqueezeModule, shapeDecoder, PPI_decoder, anchor_decoder


_EGA, _DCNResNet, _SqueezeModule, _shapeDecoder, _PPI_decoder, _anchor_decoder = _import_official()


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


def normal_01(x: torch.Tensor) -> torch.Tensor:
    """The authors' `normal_ecg_torch_01`: per-sample min-max to [0, 1]."""
    flat = x.flatten(1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    return ((flat - lo) / (hi - lo).clamp_min(1e-8)).view_as(x)


class RadarODEMTLContinuous(_EGA):
    """radarODE-MTL: shared backbone, the three published heads, EGA, plus a
    continuous head for the scored task."""

    def __init__(self, in_bins: int = 50, dropout: float = 0.1, max_epochs: int = 100,
                 ega_temp: float = 1.0, grad_checkpoint: bool = True) -> None:
        super().__init__()
        self.in_bins = in_bins
        # EGA computes one backward pass per task with the graph retained, so the
        # backbone's activations would be held four times over. Checkpointing recomputes
        # them instead, which keeps the authors' rep_grad=False setting and the protocol
        # batch size of 16 affordable on a 16 GB card. Numerically identical, ~30% slower.
        self.grad_checkpoint = grad_checkpoint
        self.backbone = _DCNResNet(in_bins)
        self.squeeze_module = _SqueezeModule(in_channels=1024, out_channels=1024)

        # Published task heads, unmodified.
        self.shape_decoder = _shapeDecoder()
        self.ppi_decoder = _PPI_decoder(output_dim=PPI_CLASSES)
        self.anchor_decoder = _anchor_decoder()

        # The added continuous head, built from the authors' own primitives.
        self.continuous_decoder = nn.Sequential(
            _DeconvBlock1d(1024, 512, kernel_size=5, stride=3, padding=1),
            _DeconvBlock1d(512, 256, kernel_size=5, stride=3, padding=1),
            _DeconvBlock1d(256, 256, kernel_size=5, stride=3, padding=1),
            _ConvBlock1d(256, 128, kernel_size=7, stride=1, padding=3),
            nn.Dropout(dropout),
            _ConvBlock1d(128, 64, kernel_size=7, stride=1, padding=3),
            _ConvBlock1d(64, 32, kernel_size=5, stride=1, padding=2),
        )
        self.head = nn.Conv1d(32, 1, kernel_size=5, padding=2)

        # EGA bookkeeping (LibMTL's AbsWeighting contract).
        self.task_name = list(TASKS)
        self.task_num = len(TASKS)
        self.rep_grad = False
        self.ega_temp = ega_temp
        self.epoch = 0
        self.train_loss_buffer = np.zeros((self.task_num, max_epochs + 1))
        self.init_param()

    # ---- AbsWeighting contract -------------------------------------------------
    def get_share_params(self):
        """The parameters EGA aligns: the shared backbone, not the task heads."""
        return chain(self.backbone.parameters(), self.squeeze_module.parameters())

    def zero_grad_share_params(self) -> None:
        self.backbone.zero_grad(set_to_none=False)
        self.squeeze_module.zero_grad(set_to_none=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ---- forward ---------------------------------------------------------------
    def represent(self, wsst: torch.Tensor) -> torch.Tensor:
        if wsst.ndim != 4 or wsst.shape[1] != self.in_bins:
            raise ValueError(f"wsst must be [B,{self.in_bins},F,T], got {tuple(wsst.shape)}")
        if self.grad_checkpoint and self.training:
            features = torch.utils.checkpoint.checkpoint(
                self.backbone, wsst, use_reentrant=False)
        else:
            features = self.backbone(wsst)
        return self.squeeze_module(features)                     # (B, 1024, T')

    def continuous(self, rep: torch.Tensor) -> torch.Tensor:
        h = self.continuous_decoder(rep)
        if h.shape[-1] != WIN:
            h = F.interpolate(h, size=WIN, mode="linear", align_corners=False)
        return self.head(h)

    def all_tasks(self, wsst: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every head's output, ready for the four task losses."""
        rep = self.represent(wsst)
        shape = self.shape_decoder(rep)
        if shape.shape[-1] != SHAPE_LEN:
            shape = F.interpolate(shape, size=SHAPE_LEN, mode="linear", align_corners=False)
        # The PPI head's linear layer is sized for the authors' 31-frame representation.
        ppi = self.ppi_decoder(F.adaptive_avg_pool1d(rep, PPI_FRAMES))
        anchor = self.anchor_decoder(rep)
        if anchor.shape[-1] != WIN:
            anchor = F.interpolate(anchor, size=WIN, mode="linear", align_corners=False)
        return {"continuous": self.continuous(rep), "ECG_shape": shape,
                "PPI": ppi, "Anchor": anchor}

    def forward(self, wsst: torch.Tensor | None = None, rcg: torch.Tensor | None = None):
        """Inference path: only the scored continuous head."""
        if wsst is None:
            raise ValueError("radarode_mtl consumes the WSST tensor")
        return self.continuous(self.represent(wsst)), None


def shape_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The authors' shapeLoss: MSE against the min-max normalised beat shape."""
    return F.mse_loss(pred, normal_01(target.clone().detach()))


def anchor_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The authors' anchorLoss: MSE against the min-max normalised anchor signal."""
    return F.mse_loss(pred, normal_01(target.clone().detach()))


def ppi_loss(logits: torch.Tensor, cycle_len: torch.Tensor) -> torch.Tensor:
    """The authors' ppiLoss.

    `cross_entropy_loss_ppi` turns the padded cycle into a one-hot at index
    `cycle_length - 1`, softmaxes it with a 1000 logit (numerically one-hot) and
    takes cross entropy against the 260-way head. `cycle_len` carries the same
    quantity, read from the fold's R peaks instead of a padding sentinel.
    """
    logits = logits.squeeze(1)
    idx = (cycle_len.long() - 1).clamp(0, PPI_CLASSES - 1)
    target = torch.zeros_like(logits)
    target[torch.arange(logits.shape[0], device=logits.device), idx] = 1000.0
    return F.cross_entropy(logits, target.softmax(dim=1))
