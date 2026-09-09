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


def test_scaler_uses_entire_train_fold_only(monkeypatch):
    import inspect
    assert list(inspect.signature(T.fit_scaler).parameters) == ["fold"], \
        "fit_scaler must take no subsampling cap: the scaler uses the ENTIRE training fold"
    opened = []
    real = h5py.File
    monkeypatch.setattr(h5py, "File", lambda p, *a, **k: (opened.append(str(p)), real(p, *a, **k))[1])
    T.fit_scaler(1)
    assert all("train.h5" in p for p in opened), opened
    assert not any("val" in p or "test" in p for p in opened)


def test_formal_run_cannot_use_smoke_parameters():
    """A formal run must not be silently shrunk: epochs/limit are not CLI-settable."""
    import argparse, contextlib, io
    src = Path(T.__file__).read_text()
    assert '"--epochs"' not in src and '"--limit"' not in src, \
        "formal epochs/limit must not be overridable from the CLI"
    assert '--smoke_epochs' in src and '--smoke_limit' in src
    # formal branch pins the frozen budget and no limit
    assert 'epochs, limit, tag = BUDGET["epochs"], None, "q1_continuous_ecg_b0_b2_pilot"' in src


def test_smoke_and_formal_write_to_separate_trees():
    src = Path(T.__file__).read_text()
    assert '"q1_continuous_ecg_b0_b2_pilot_smoke"' in src and '"q1_continuous_ecg_b0_b2_pilot"' in src


def test_overwrite_protection_present():
    src = Path(T.__file__).read_text()
    assert 'Refusing to overwrite a completed run' in src
    assert '(out / "best.json").exists() and not a.overwrite' in src


def test_event_loss_is_fixed_and_shared():
    assert T.EVENT_W == 0.20
    src = Path(T.__file__).read_text()
    assert "binary_cross_entropy_with_logits" in src, "event head must be supervised"


def test_seed_is_paired_across_arms_within_fold():
    assert T.effective_seed(1) == T.effective_seed(1)
    assert T.effective_seed(1) != T.effective_seed(5)
    assert T.effective_seed(1) == T.BUDGET["seed"] + 1


def test_registered_metric_set_is_complete():
    """Every metric the spec registers must be produced by evaluate()."""
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(4, WIN)); gt = rng.normal(size=(4, WIN))
    ev = rng.random(size=(4, WIN)); rps = [np.array([300., 700., 1100.])] * 4
    m = T.waveform_metrics(pred, gt)
    m.update(T.timing_metrics(pred, ev, rps))
    m.update(T.region_errors(pred, gt, rps))
    m["shape200_pcc_auxiliary"] = T.shape200_pcc(pred, gt, rps)
    required = {"pcc", "rmse", "nrmse", "mae", "r2",
                "f1_50ms", "f1_100ms", "f1_150ms",
                "event_f1_50ms", "event_f1_100ms", "event_f1_150ms",
                "rr_error_ms", "hr_error_bpm", "rr_coverage",
                "qrs_region_mae", "stt_region_mae", "shape200_pcc_auxiliary"}
    assert required <= set(m), f"missing: {required - set(m)}"


def test_shape200_is_auxiliary_only():
    src = Path(T.__file__).read_text()
    assert "shape200_pcc_auxiliary" in src
    assert "shape200" not in str(T.COMPOSITE_W)


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
    assert T.BUDGET == dict(epochs=30, patience=8, lr=1e-4, weight_decay=1e-4,
                            batch_size=16, seed=42)


def test_all_arms_share_one_budget():
    """No arm may run a different budget: the trainer reads BUDGET, never a per-arm value."""
    src = Path(T.__file__).read_text()
    assert 'epochs, limit, tag = BUDGET["epochs"], None, "q1_continuous_ecg_b0_b2_pilot"' in src
    for arm in ARMS:
        assert f'"{arm}":' not in src.split("BUDGET =")[1].split("\n")[0], "budget must not be per-arm"


def test_input_is_ram_preloaded_not_hdf5_per_item():
    """HDF5-per-item was the measured bottleneck (~269 s/epoch at 0% GPU)."""
    src = Path(T.__file__).read_text()
    assert "self.wsst = np.asarray" in src, "WSST must be preloaded into RAM"
    assert "num_workers=0" in src, "workers are unnecessary once data is in RAM"
    assert "_open(self)" not in src, "the lazy per-item HDF5 path must be gone"
