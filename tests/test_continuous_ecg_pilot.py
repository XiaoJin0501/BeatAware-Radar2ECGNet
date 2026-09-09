"""Invariants for q1_continuous_ecg_b0_b2_pilot. Pure checks; no training, no test split."""
from __future__ import annotations
import json, sys
from pathlib import Path
import h5py, numpy as np, pytest, torch, torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.models.continuous.ce_net import CENet, param_report, WIN  # noqa: E402
import scripts.train_continuous_ecg_pilot as T  # noqa: E402

ARMS = ["B0", "B1", "B2a", "B2a-wide", "B2b"]
DATA = Path("/home/qhh2237/Datasets/MMECG/processed/loso")


@pytest.fixture(scope="module")
def models(): return {a: CENet(a) for a in ARMS}


@pytest.mark.parametrize("arm", ARMS)
def test_output_is_full_window(models, arm):
    x = torch.randn(2, 50, 71, 240)
    kw = {"rcg": torch.randn(2, 50, WIN)} if arm == "B2b" else {}
    y, ev = models[arm](x, **kw)
    assert y.shape == (2, 1, WIN) and ev.shape == (2, 2, WIN)


def test_decoder_parity_across_b2_arms(models):
    d = {a: param_report(models[a])["per_module"]["decoder"] for a in ("B2a", "B2a-wide", "B2b")}
    assert len(set(d.values())) == 1, f"decoder must be identical across B2 arms: {d}"


def test_b2a_wide_matches_b2b_capacity(models):
    a, b = param_report(models["B2a-wide"])["total"], param_report(models["B2b"])["total"]
    assert abs(a - b) / b < 0.001, f"B2a-wide {a} must match B2b {b} within 0.1%"


def test_b2b_delta_is_only_stem_and_fusion(models):
    pa, pb = param_report(models["B2a"])["per_module"], param_report(models["B2b"])["per_module"]
    delta = param_report(models["B2b"])["total"] - param_report(models["B2a"])["total"]
    assert delta == pb["rcg_stem"] + (pb["fuse"] - pa["fuse"])
    assert pa["wsst_stem"] == pb["wsst_stem"]


def test_output_range_b0_bounded_others_linear(models):
    """Behavioural, not structural: LegacyDecoder applies torch.sigmoid functionally,
    so an isinstance(nn.Sigmoid) check cannot see it. What matters is the range."""
    torch.manual_seed(0)
    x = torch.randn(8, 50, 71, 240) * 5.0            # large drive to expose saturation
    for a in ARMS:
        kw = {"rcg": torch.randn(8, 50, WIN) * 5.0} if a == "B2b" else {}
        with torch.no_grad():
            y, _ = models[a](x, **kw)
        if a == "B0":
            assert y.min() >= 0.0 and y.max() <= 1.0, "B0 must stay bounded in [0,1]"
        else:
            assert not (y.min() >= 0.0 and y.max() <= 1.0) or y.std() > 0, \
                f"{a} must have an unbounded linear head"
            head = models[a].decoder.out if hasattr(models[a].decoder, "out") else models[a].decoder.head
            assert isinstance(head if isinstance(head, nn.Conv1d) else head[-1], nn.Conv1d), \
                f"{a} output head must terminate in a linear Conv1d"


def test_b2b_requires_rcg(models):
    with pytest.raises(ValueError):
        models["B2b"](torch.randn(2, 50, 71, 240))


def test_composite_ignores_shape200():
    assert set(T.COMPOSITE_W) == {"pcc", "nrmse", "r2", "f1"}
    m = {"pcc": 0.5, "nrmse": 0.5, "r2": 0.5, "f1_100ms": 0.5, "shape200_pcc": 9.9}
    assert abs(T.composite(m) - (0.4*0.5 + 0.25*0.5 + 0.20*0.5 + 0.15*0.5)) < 1e-12


def test_dataset_refuses_test_split():
    with pytest.raises(AssertionError):
        T.Fold(1, "test", "B1", (0.0, 1.0))


def test_no_test_wsst8s_exists():
    assert list(DATA.glob("fold_*/test_wsst8s.h5")) == []


@pytest.mark.parametrize("fold,split", [(1, "train"), (1, "val"), (5, "val"), (9, "val")])
def test_wsst_declares_full_support(fold, split):
    with h5py.File(DATA / f"fold_{fold:02d}" / f"{split}_wsst8s.h5", "r") as h:
        a, s = dict(h.attrs), h["wsst"].shape
    assert a["support"] == "full_window_samples_0_1600"
    assert float(a["supported_fraction_of_target"]) == 1.0
    assert (s[2], s[3]) == (71, 240)
    assert abs(float(a["hop_ms"]) - 100/3) < 1e-3


def test_scaler_uses_train_only(monkeypatch):
    opened = []
    real = h5py.File
    monkeypatch.setattr(h5py, "File", lambda p, *a, **k: (opened.append(str(p)), real(p, *a, **k))[1])
    T.fit_scaler(1, cap=8)
    assert all("train.h5" in p for p in opened), opened
    assert not any("val" in p or "test" in p for p in opened)


def test_metrics_are_affine_consistent():
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(3, 400)); pred = gt + 0.1 * rng.normal(size=(3, 400))
    m1 = T.waveform_metrics(pred, gt)
    m2 = T.waveform_metrics(pred * 3.0, gt * 3.0)          # common affine scale
    for k in ("pcc", "nrmse", "r2"):
        assert abs(m1[k] - m2[k]) < 1e-8, f"{k} must be affine-invariant"
    assert abs(m2["rmse"] / m1["rmse"] - 3.0) < 1e-6, "rmse must scale with units"


def test_legacy_arm_set_is_b0_only():
    assert T.LEGACY_TARGET_ARMS == ("B0",)


def test_budget_is_shared_and_frozen():
    assert T.BUDGET == dict(epochs=80, patience=20, lr=1e-4, weight_decay=1e-4,
                            batch_size=16, seed=42)
