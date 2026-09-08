"""Training-only supervision linking a waveform cycle to its morphology head."""

import torch
import torch.nn.functional as F


def morphology_consistency_loss(
    waveform: torch.Tensor,
    shape: torch.Tensor,
    reference: torch.Tensor,
    cycle_bounds: torch.Tensor,
    target_mode: str = "head",
    cycle_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Use reference cycle coordinates only in training; return loss and coverage.

    Reference amplitude statistics put both predictions in the same coordinate
    system without independently normalizing away waveform amplitude errors.
    The head target is detached. Shared encoder parameters still receive the
    waveform gradient, so this is not a frozen external teacher.
    """
    if target_mode not in {"head", "reference"}:
        raise ValueError("target_mode must be head or reference")
    if waveform.ndim != 3 or waveform.shape[1] != 1:
        raise ValueError("waveform must have shape (batch, 1, time)")
    if reference.shape != waveform.shape:
        raise ValueError("reference must match waveform")
    if shape.ndim != 3 or shape.shape[:2] != waveform.shape[:2]:
        raise ValueError("shape must have shape (batch, 1, morphology_time)")
    if shape.shape[-1] < 2 or cycle_bounds.shape != (waveform.shape[0], 2):
        raise ValueError("invalid morphology length or cycle_bounds")
    if cycle_valid is not None and cycle_valid.shape != (waveform.shape[0],):
        raise ValueError("cycle_valid must have one flag per waveform")

    terms = []
    for i, (start, end) in enumerate(cycle_bounds.detach().cpu().tolist()):
        if cycle_valid is not None and not bool(cycle_valid[i]):
            continue
        if int(start) != start or int(end) != end:
            raise ValueError("cycle boundaries must be integer sample indices")
        start, end = int(start), int(end)
        if start < 0 or end > waveform.shape[-1] or end - start <= 10:
            continue
        pred = F.interpolate(waveform[i:i + 1, :, start:end],
                             size=shape.shape[-1], mode="linear", align_corners=False)
        gt = F.interpolate(reference[i:i + 1, :, start:end].detach(),
                           size=shape.shape[-1], mode="linear", align_corners=False)
        offset = gt.amin(dim=-1, keepdim=True)
        scale = gt.amax(dim=-1, keepdim=True) - offset
        if not torch.isfinite(gt).all() or scale.item() <= 1e-4:
            continue
        target = shape[i:i + 1].detach() if target_mode == "head" else (gt - offset) / scale
        terms.append(F.l1_loss((pred - offset) / scale, target))
    if not terms:
        return waveform.sum() * 0.0, 0
    return torch.stack(terms).mean(), len(terms)
