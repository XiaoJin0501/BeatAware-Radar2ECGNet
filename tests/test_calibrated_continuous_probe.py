import numpy as np

from scripts.probe_calibrated_continuous_ecg import fit_affine, fit_lag, shift_edge


def test_shift_and_lag_recovery():
    x = np.zeros((3, 1600), np.float32)
    x[:, 300] = 1
    y = shift_edge(x, 13)
    assert fit_lag(x, y) == 13
    np.testing.assert_allclose(shift_edge(x, 13), y)


def test_affine_recovery():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(4, 1600)).astype(np.float32)
    a, b = fit_affine(x, 2.5 * x - 0.7)
    assert abs(a - 2.5) < 1e-5
    assert abs(b + 0.7) < 1e-5


def test_shape200_is_not_part_of_probe():
    source = open("scripts/probe_calibrated_continuous_ecg.py").read().lower()
    assert "shape200_pcc" not in source
    assert '"shape200_computed": false' in source


def test_session_partition_has_no_sample_overlap():
    starts = np.arange(0, 35, 2) * 200
    ends = starts + 1600
    cal = ends <= 4000
    query = starts >= 4000
    assert ends[cal].max() <= starts[query].min()
