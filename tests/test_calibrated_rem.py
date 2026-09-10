import torch

from src.models.continuous.calibrated_rem import CalibrationConditionedREM, interaction_contrast


def _inputs():
    return (torch.randn(3, 1, 1600), torch.randn(3, 2, 1600),
            torch.randn(7, 1, 1600), torch.randn(7, 1, 1600))


def test_output_is_continuous_and_identity_initialized():
    model = CalibrationConditionedREM()
    query, events, cp, cg = _inputs()
    output, aux = model(query, events, cp, cg)
    assert output.shape == (3, 1, 1600)
    torch.testing.assert_close(output, query)
    assert aux["residual"].shape == query.shape


def test_factorial_switches_are_parameter_matched():
    counts = []
    for event in (False, True):
        for morph in (False, True):
            model = CalibrationConditionedREM(use_event=event, use_morphology=morph)
            counts.append(sum(p.numel() for p in model.parameters()))
    assert len(set(counts)) == 1


def test_morphology_off_zeroes_token_and_event_off_sets_unit_gate():
    query, events, cp, cg = _inputs()
    model = CalibrationConditionedREM(use_event=False, use_morphology=False)
    _, aux = model(query, events, cp, cg)
    assert torch.count_nonzero(aux["calibration_token"]) == 0
    assert torch.all(aux["event_gate"] == 1)


def test_interaction_contrast():
    scores = {"neither": 0.4, "event_only": 0.5, "morphology_only": 0.55, "full": 0.7}
    assert abs(interaction_contrast(scores) - 0.05) < 1e-9
