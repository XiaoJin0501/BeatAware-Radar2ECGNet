import inspect

import torch

from scripts.eval_tta_pilot import (
    WIN,
    adapt_subject_shift,
    cyclic_prominence_loss,
    fractional_circular_shift,
)


def test_fractional_shift_has_expected_integer_direction():
    x = torch.zeros(1, 1, WIN)
    x[..., 100] = 1.0
    shifted = fractional_circular_shift(x, torch.tensor(5.0))
    assert int(shifted.argmax()) == 105


def test_cyclic_prominence_is_shift_invariant():
    torch.manual_seed(7)
    x = torch.randn(2, 1, WIN)
    a = cyclic_prominence_loss(x)
    b = cyclic_prominence_loss(fractional_circular_shift(x, torch.tensor(7.25)))
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_adaptation_api_cannot_receive_ground_truth():
    parameters = inspect.signature(adapt_subject_shift).parameters
    assert set(parameters) == {"model", "calibration_wsst", "steps", "lr"}
