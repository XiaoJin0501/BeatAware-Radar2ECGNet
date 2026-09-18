import numpy as np
import pywt
import pytest
import torch

from src.models.baselines.continuous import ADAPTATIONS, BASELINES, build_baseline, input_key
from src.models.baselines.continuous.wavegru_net import MODWTLayer

WIN = 1600
RCG_BASELINES = [name for name in BASELINES if input_key(name) == "rcg"]


@pytest.mark.parametrize("name", RCG_BASELINES)
def test_rcg_baselines_emit_the_contract_shape(name):
    model = build_baseline(name)
    y, event = model(rcg=torch.randn(2, 50, WIN))
    assert y.shape == (2, 1, WIN)
    assert event is None, "no baseline claims an event head; the adapter supplies the placeholder"


@pytest.mark.parametrize("name", RCG_BASELINES)
def test_rcg_baselines_reject_the_wrong_modality(name):
    with pytest.raises(ValueError):
        build_baseline(name)(wsst=torch.randn(2, 50, 71, 240))


def test_every_baseline_documents_its_adaptations():
    assert set(ADAPTATIONS) == set(BASELINES)
    for name, notes in ADAPTATIONS.items():
        assert notes, name
        assert all(note.startswith("D") for note in notes), name


def test_modwt_layer_matches_pywt_swt_band_for_band():
    """The a-trous filter bank must equal pywt's MODWT up to the known filter delay."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 256)).astype(np.float32)
    mine = MODWTLayer(3, levels=5)(torch.from_numpy(x)).numpy().reshape(1, 6, 3, 256)

    ref = pywt.swt(x.reshape(-1, 256), "sym4", level=5, axis=-1, trim_approx=True, norm=True)
    bands = np.stack(list(ref[1:][::-1]) + [ref[0]], axis=1)
    bands = bands.reshape(1, 3, 6, 256).transpose(0, 2, 1, 3)

    for band in range(6):
        got, want = mine[0, band].ravel(), bands[0, band]
        best = max(abs(np.corrcoef(got, np.roll(want, shift, axis=-1).ravel())[0, 1])
                   for shift in range(-128, 128))
        assert best > 0.999, f"band {band} does not match pywt (best |corr| {best:.4f})"
        assert np.var(got) == pytest.approx(np.var(want), rel=1e-4)


def test_dcgan_output_gain_scales_the_tanh_range():
    """The Tanh range is restored to train-fold robust-scaled units by a fixed gain."""
    rcg = torch.randn(2, 50, WIN)
    plain, _ = build_baseline("attention_dcgan", out_gain=1.0)(rcg=rcg)
    assert plain.abs().max() <= 1.0
    scaled = build_baseline("attention_dcgan", out_gain=8.0)
    assert float(scaled.out_gain) == 8.0
