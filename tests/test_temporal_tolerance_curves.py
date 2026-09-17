import numpy as np

from scripts.eval_temporal_tolerance_curves import lag_scores, tolerance_curve


def test_fixed_support_lag_recovery_and_monotonic_curve():
    rng = np.random.default_rng(7)
    target = rng.normal(size=(4, 1600))
    pred = np.empty_like(target)
    pred[:, 6:] = target[:, :-6]
    pred[:, :6] = 0
    lags, scores = lag_scores(pred, target, max_lag=20)
    curve, best = tolerance_curve(lags, scores)
    assert np.all(best == 6)
    values = np.asarray([x["mean_ncc"] for x in curve])
    assert np.all(np.diff(values) >= -1e-12)
    assert values[3] > 0.999


def test_lag_scores_reject_shape_mismatch():
    with np.testing.assert_raises(ValueError):
        lag_scores(np.zeros((2, 1600)), np.zeros((3, 1600)))
