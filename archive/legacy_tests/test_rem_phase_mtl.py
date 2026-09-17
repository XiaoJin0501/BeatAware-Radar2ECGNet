import torch

from configs.mmecg_config import MMECGConfig
from src.models.model_factory import build_mmecg_model
from src.models.remalign import REMPhaseMTLNet, SoftCardiacPhaseCanonicalizer


def _peaked_mask(batch: int = 2, length: int = 1600) -> torch.Tensor:
    mask = torch.full((batch, 1, length), 1e-3)
    mask[:, :, 600] = 0.99
    mask[:, :, 980] = 0.99
    return mask


def test_phase_canonicalizer_recovers_predicted_cycle_bounds():
    canonicalizer = SoftCardiacPhaseCanonicalizer(temperature=0.02)
    h = torch.randn(2, 16, 400, requires_grad=True)
    r = _peaked_mask()
    qrs = torch.zeros_like(r)
    out, events, bounds = canonicalizer(h, r, qrs)
    assert out.shape == (2, 16, 200)
    assert events.shape == (2, 2, 200)
    assert torch.allclose(bounds[:, 0], torch.full((2,), 600.0), atol=1.0)
    assert torch.allclose(bounds[:, 1], torch.full((2,), 980.0), atol=1.0)
    out.mean().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_rem_phase_mtl_output_contract_and_interaction_switches():
    x = torch.randn(2, 50, 71, 120)
    for canonicalize in (False, True):
        for condition in (False, True):
            model = REMPhaseMTLNet(
                hidden=32,
                use_phase_canonicalizer=canonicalize,
                use_event_conditioning=condition,
            ).eval()
            with torch.no_grad():
                ecg, peaks = model(x)
            assert ecg.shape == (2, 1, 1600)
            assert ecg.shape200_pred.shape == (2, 1, 200)
            assert ecg.phase_bounds.shape == (2, 2)
            assert len(peaks) == 3 and peaks[0].shape == (2, 1, 1600)
            if not condition:
                assert torch.count_nonzero(ecg.phase_events) == 0


def test_model_factory_builds_rem_phase_mtl():
    cfg = MMECGConfig()
    cfg.model_arch = "rem_phase_mtl"
    cfg.baseline_hidden = 32
    model = build_mmecg_model(cfg)
    assert isinstance(model, REMPhaseMTLNet)
