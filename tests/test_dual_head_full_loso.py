from scripts.run_dual_head_full_loso import FOLDS, REUSED


def test_full_loso_fold_plan_is_complete_and_reuse_is_bounded():
    assert FOLDS == tuple(range(1, 12))
    assert REUSED == (2, 6, 10)
    assert set(REUSED) < set(FOLDS)
