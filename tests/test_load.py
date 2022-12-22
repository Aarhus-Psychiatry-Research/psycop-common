"""Testing of loader functions."""

from psycop_model_training.data_loader.utils import load_and_filter_train_from_cfg
from psycop_model_training.utils.config_schemas.full_config import FullConfigSchema


def test_load_lookbehind_exceeds_lookbehind_threshold(
    muteable_test_config: FullConfigSchema,
):
    """Test that columns are dropped if their lookbehind are larger than the
    lookbehind threshold."""
    cfg = muteable_test_config

    n_cols_before_filtering = load_and_filter_train_from_cfg(cfg=cfg).shape[1]

    cfg.preprocessing.pre_split.lookbehind_combination = [30, 60]

    n_cols_after_filtering = load_and_filter_train_from_cfg(cfg=cfg).shape[1]

    assert n_cols_before_filtering - n_cols_after_filtering == 2


def test_load_lookbehind_not_in_lookbehind_combination(
    muteable_test_config: FullConfigSchema,
):
    """Test that columns are dropped if their lookbehind is not in the
    specified lookbehind combination list."""
    cfg = muteable_test_config

    n_cols_before_filtering = load_and_filter_train_from_cfg(cfg=cfg).shape[1]

    cfg.preprocessing.pre_split.lookbehind_combination = [60]

    n_cols_after_filtering = load_and_filter_train_from_cfg(cfg=cfg).shape[1]

    assert n_cols_before_filtering - n_cols_after_filtering == 3
