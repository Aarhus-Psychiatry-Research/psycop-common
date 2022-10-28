"""Testing of loader functions."""

from psycopt2d.load import load_train_from_cfg
from psycopt2d.utils.config_schemas import FullConfigSchema


def test_load_lookbehind_exceeds_lookbehind_threshold(
    muteable_test_config: FullConfigSchema,
):
    """Test that columns are dropped if their lookbehind are larger than the
    lookbehind threshold."""
    cfg = muteable_test_config

    n_cols_before_filtering = load_train_from_cfg(cfg=cfg).shape[1]

    cfg.data.min_lookbehind_days = 60

    n_cols_after_filtering = load_train_from_cfg(cfg=cfg).shape[1]

    assert n_cols_before_filtering - n_cols_after_filtering == 2


def test_load_lookbehind_not_in_lookbehind_combination(
    muteable_test_config: FullConfigSchema,
):
    """Test that columns are dropped if their lookbehind is not in the
    specified lookbehind combination list."""
    cfg = muteable_test_config

    n_cols_before_filtering = load_train_from_cfg(cfg=cfg).shape[1]

    cfg.data.lookbehind_combination = [60]

    n_cols_after_filtering = load_train_from_cfg(cfg=cfg).shape[1]

    assert n_cols_before_filtering - n_cols_after_filtering == 3
