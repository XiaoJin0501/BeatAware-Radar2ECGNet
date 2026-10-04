import pytest
import torch

from src.models.continuous.dual_head_rem import DualHeadREM


@pytest.mark.parametrize("mode,channels", [("r_multiscale", 2), ("qrst", 4)])
def test_dual_head_rem_forward_backward(mode, channels):
    model = DualHeadREM(anchor_mode=mode, in_bins=3, width=16,
                        head_width=8, dropout=0.0)
    output = model(torch.randn(2, 3, 11, 24))
    assert output["shape"].shape == (2, 1, 200)
    assert output["anchor_logits"].shape == (2, channels, 1600)
    loss = output["shape"].mean() + output["anchor_logits"].square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)


def test_dual_head_rem_rejects_unknown_anchor_mode():
    with pytest.raises(ValueError, match="anchor_mode"):
        DualHeadREM(anchor_mode="unknown")


def test_control_and_candidate_are_exactly_capacity_matched():
    control = DualHeadREM(anchor_mode="r_multiscale", in_bins=3, width=16,
                          head_width=8, dropout=0.0)
    candidate = DualHeadREM(anchor_mode="qrst", in_bins=3, width=16,
                            head_width=8, dropout=0.0)
    assert sum(p.numel() for p in control.parameters()) == sum(
        p.numel() for p in candidate.parameters())
