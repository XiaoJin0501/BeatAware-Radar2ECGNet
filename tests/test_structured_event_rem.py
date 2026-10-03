import pytest
import torch

from src.models.continuous.structured_event_rem import FIDUCIALS, StructuredEventREM


def test_structured_event_rem_forward_and_backward():
    model = StructuredEventREM(in_bins=3, width=16, head_width=8, dropout=0.0)
    x = torch.randn(2, 3, 11, 24)
    logits = model(x)
    assert logits.shape == (2, len(FIDUCIALS), 1600)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.rand_like(logits))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)


def test_structured_event_rem_rejects_wrong_range_count():
    model = StructuredEventREM(in_bins=3, width=16, head_width=8)
    with pytest.raises(ValueError, match="wsst must be"):
        model(torch.randn(1, 2, 11, 24))
