"""Regression tests for the registered ablation ladder (q1_remnet_ablation_ladder)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_continuous_ecg_pilot import (  # noqa: E402
    EVENT_W, LADDER_RUNGS, LADDER_TAG, is_registered_final_mode)
from src.models.continuous.ce_net import CENet  # noqa: E402

SHARED = ("wsst_stem", "decoder.tcn", "rcg_stem")


def _shared_grad_from_event(model, x, rcg=None):
    model.train()
    _, ev = model(x, rcg=rcg)
    ev.sum().backward()
    return sum(int(p.grad is not None and p.grad.abs().sum() > 0)
               for n, p in model.named_parameters() if n.startswith(SHARED))


def test_default_b1_is_unchanged():
    torch.manual_seed(0); a = CENet("B1")
    torch.manual_seed(0); b = CENet("B1", event_stop_grad=False)
    for p, q in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.equal(p, q)
    x = torch.randn(2, 50, 71, 240)
    a.eval(); b.eval()
    assert torch.equal(a(x)[0], b(x)[0]) and torch.equal(a(x)[1], b(x)[1])


def test_stop_grad_blocks_event_gradient_into_backbone():
    x = torch.randn(2, 50, 71, 240)
    assert _shared_grad_from_event(CENet("B1", event_stop_grad=True), x) == 0
    assert _shared_grad_from_event(CENet("B1"), x) > 0


def test_stop_grad_keeps_waveform_gradient():
    m = CENet("B1", event_stop_grad=True); m.train()
    y, _ = m(torch.randn(2, 50, 71, 240))
    y.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in m.named_parameters() if n.startswith("wsst_stem"))


def test_r0_rcg_shapes_and_no_wsst_stem():
    m = CENet("R0-rcg")
    y, ev = m(None, rcg=torch.randn(2, 50, 1600))
    assert tuple(y.shape) == (2, 1, 1600) and tuple(ev.shape) == (2, 2, 1600)
    assert not hasattr(m, "wsst_stem")


def test_rungs_change_one_variable_each():
    r = LADDER_RUNGS
    assert r["R0"]["model_arm"] == "R0-rcg" and r["R1"]["model_arm"] == "B1"
    assert r["R0"]["event_w"] == r["R1"]["event_w"] == 0.0
    assert r["R2"]["event_w"] == r["R3"]["event_w"] == EVENT_W
    assert r["R2"]["event_stop_grad"] and not r["R3"]["event_stop_grad"]


def test_final_mode_gate():
    assert is_registered_final_mode(True, False, "B1", "L3", LADDER_TAG, 42, "mmecg", "R0")
    assert not is_registered_final_mode(True, False, "B1", "L3", LADDER_TAG, 123, "mmecg", "R0")
    assert not is_registered_final_mode(True, False, "B1", "L3", "other_tag", 42, "mmecg", "R0")
    assert not is_registered_final_mode(True, True, "B1", "L3", LADDER_TAG, 42, "mmecg", "R0")
