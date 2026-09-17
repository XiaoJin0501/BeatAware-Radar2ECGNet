from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.train_continuous_ecg_pilot import (
    EVENT_W,
    LOSS_WEIGHTS,
    MRSTFT_CONFIG,
    auxiliary_waveform_loss,
)
from src.losses.losses import MultiResolutionSTFTLoss
from src.models.continuous.ce_net import (
    CENet,
    param_report,
    reset_module_parameters_paired,
)


def _model(arm: str, seed: int = 43) -> CENet:
    torch.manual_seed(seed)
    model = CENet(arm)
    reset_module_parameters_paired(model.decoder, seed + 10_000)
    return model


def test_capacity_and_decoder_parity() -> None:
    control = _model("B2a-wide")
    dual = _model("B2b")
    assert param_report(control)["total"] == 1_487_551
    assert param_report(dual)["total"] == 1_487_667
    assert (param_report(dual)["total"] - param_report(control)["total"]) == 116
    left, right = control.decoder.state_dict(), dual.decoder.state_dict()
    assert left.keys() == right.keys()
    assert all(left[k].shape == right[k].shape for k in left)
    assert all(torch.equal(left[k], right[k]) for k in left)


def test_locked_l3_configuration() -> None:
    assert EVENT_W == 0.20
    assert LOSS_WEIGHTS["L3"] == 0.50
    assert MRSTFT_CONFIG == {
        "fft_sizes": [128, 256, 512],
        "hop_sizes": [32, 64, 128],
        "win_lengths": [128, 256, 512],
    }


def test_b2_forward_backward_shapes_and_finite_gradients() -> None:
    batch = 2
    wsst = torch.randn(batch, 50, 71, 240)
    rcg = torch.randn(batch, 50, 1600)
    target = torch.randn(batch, 1, 1600)
    event_target = torch.rand(batch, 2, 1600)
    mrstft = MultiResolutionSTFTLoss(**MRSTFT_CONFIG)

    for arm in ("B2a-wide", "B2b"):
        model = _model(arm)
        pred, event = model(wsst, rcg=rcg if arm == "B2b" else None)
        assert pred.shape == (batch, 1, 1600)
        assert event.shape == (batch, 2, 1600)
        loss = (F.huber_loss(pred, target)
                + EVENT_W * F.binary_cross_entropy_with_logits(event, event_target)
                + auxiliary_waveform_loss("L3", pred, target, mrstft))
        loss.backward()
        assert torch.isfinite(loss)
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert grads and all(g is not None for g in grads)
        assert all(torch.isfinite(g).all() for g in grads)


def test_b2b_rejects_shape_mismatch_without_broadcasting() -> None:
    model = _model("B2b")
    wsst = torch.randn(2, 50, 71, 240)
    bad_rcg = torch.randn(2, 50, 1599)
    try:
        model(wsst, bad_rcg)
    except ValueError as exc:
        assert "rcg must have shape" in str(exc)
    else:
        raise AssertionError("B2b accepted a misaligned RCG tensor")
