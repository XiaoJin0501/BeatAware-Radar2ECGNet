import numpy as np

from scripts.eval_60s_calibration_loso import (
    BOUNDARY,
    chronological_masks,
    estimate_calibration,
    shifted,
)


def test_chronological_masks_have_no_source_overlap():
    record = np.asarray([b"a"] * 31 + [b"b"] * 2)
    start = np.asarray(list(range(0, 31 * 400, 400)) + [0, 400])
    end = start + 1600
    calibration, evaluation, excluded, first = chronological_masks(record, start, end)
    assert first == "a"
    assert np.all(end[calibration] <= BOUNDARY)
    assert np.all(start[evaluation & (record == b"a")] >= BOUNDARY)
    assert not np.any(calibration & evaluation)
    assert excluded.sum() == 3


def test_calibration_recovers_known_integer_lag_and_scale():
    rng = np.random.default_rng(4)
    target = rng.normal(size=(10, 1600)).astype(np.float32)
    prediction = shifted(target, -11) / 2.5
    lag, alpha, _ = estimate_calibration(prediction, target)
    assert lag == 11
    assert abs(alpha - 2.5) < 0.02
