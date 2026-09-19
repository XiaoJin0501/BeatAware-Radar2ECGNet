import pytest

from scripts.train_continuous_ecg_pilot import (
    DATASETS, _set_split_root, final_continuous_tag, is_registered_final_mode,
)
from scripts.train_continuous_baseline import registered_tag


def test_crrvs_shape_build_is_refused_as_continuous_ground_truth():
    """The Shape-200 build min-max normalises each window; it may not train Tier 2."""
    with pytest.raises(SystemExit):
        _set_split_root("q1_shape_kfold", "crrvs")


def test_mmecg_broken_split_is_still_refused():
    with pytest.raises(SystemExit):
        _set_split_root("loso", "mmecg")


def test_registered_tags_are_dataset_specific():
    assert final_continuous_tag(42) == "q1_remnet_continuous_strict_loso_seed42"
    assert final_continuous_tag(42, "crrvs") == "q1_remnet_continuous_crrvs_seed42"
    assert registered_tag(123, "crrvs") == "q1_matched_baselines_continuous_crrvs_seed123"
    assert is_registered_final_mode(True, False, "B1", "L3",
                                    "q1_remnet_continuous_crrvs_seed42", 42, "crrvs")
    assert not is_registered_final_mode(True, False, "B1", "L3",
                                        "q1_remnet_continuous_strict_loso_seed42", 42, "crrvs")


def test_dataset_geometry():
    assert DATASETS["mmecg"]["in_bins"] == 50 and DATASETS["mmecg"]["folds"] == 11
    assert DATASETS["crrvs"]["in_bins"] == 1 and DATASETS["crrvs"]["folds"] == 5
