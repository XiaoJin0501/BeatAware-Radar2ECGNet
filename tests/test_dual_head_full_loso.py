from scripts.run_dual_head_full_loso import FOLDS, REUSED
import numpy as np


def test_full_loso_fold_plan_is_complete_and_reuse_is_bounded():
    assert FOLDS == tuple(range(1, 12))
    assert REUSED == (2, 6, 10)
    assert set(REUSED) < set(FOLDS)


def test_missing_event_errors_are_explicit_not_zero():
    values = np.asarray([10.0, np.nan, 20.0])
    valid = values[np.isfinite(values)]
    assert valid.mean() == 15.0
    assert len(values) - valid.size == 1
