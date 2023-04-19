"""Test custom preprocessing steps."""
from psycop_model_training.config_schemas.debug import DebugConfSchema
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.data_loader.utils import load_and_filter_train_from_cfg


def test_drop_datetime_predictor_columns(
    muteable_test_config: FullConfigSchema,
):
    """Test that columns are dropped if their lookbehind is not in the
    specified lookbehind combination list."""
    cfg = muteable_test_config

    cfg.preprocessing.pre_split.drop_datetime_predictor_columns = True
    cfg.preprocessing.post_split.imputation_method = None
    cfg.preprocessing.post_split.feature_selection.name = None
    cfg.preprocessing.post_split.scaling = None
    cfg.data.pred_prefix = "timestamp"

    train_df = load_and_filter_train_from_cfg(cfg=cfg)

    assert len([x for x in train_df.columns if "timestamp" in x]) == 0


def test_drop_rows_with_insufficient_lookahead(
    muteable_test_config: FullConfigSchema,
):
    """Test that rows are dropped if they can't look sufficiently far into the
    correct direction."""
    cfg = muteable_test_config
    cfg.preprocessing.pre_split.keep_only_one_outcome_col = False
    cfg.debug = DebugConfSchema(assert_outcome_col_matching_lookahead_exists=False)

    # No lookahead
    cfg.preprocessing.pre_split.min_lookahead_days = 0
    df_no_min_lookahead = load_and_filter_train_from_cfg(cfg=cfg)
    max_timestamp_sans_lookahead = df_no_min_lookahead["timestamp"].max()

    # With lookahead
    cfg.preprocessing.pre_split.min_lookahead_days = 365
    df_with_min_lookahead = load_and_filter_train_from_cfg(cfg=cfg)
    max_timestamp_with_lookahead = df_with_min_lookahead["timestamp"].max()

    # Get difference between max_timestamp and max_timestamp_with_lookahead in days
    assert (
        max_timestamp_sans_lookahead - max_timestamp_with_lookahead
    ).days >= cfg.preprocessing.pre_split.min_lookahead_days


def test_drop_rows_with_insufficient_lookbehind(
    muteable_test_config: FullConfigSchema,
):
    """Test that rows are dropped if they can't look sufficiently far into the
    correct direction."""
    cfg = muteable_test_config

    # Without lookbehind
    min_lookbehind = 30
    cfg.preprocessing.pre_split.lookbehind_combination = [min_lookbehind]
    df_no_lookbehind = load_and_filter_train_from_cfg(cfg=cfg)
    min_timestamp_sans_lookahead = df_no_lookbehind["timestamp"].min()

    # With lookbehind
    cfg.preprocessing.pre_split.lookbehind_combination = [100]
    df_with_lookbehind = load_and_filter_train_from_cfg(cfg=cfg)
    min_timestamp_with_lookahead = df_with_lookbehind["timestamp"].min()

    # Get difference between max_timestamp and max_timestamp_with_lookahead in days
    assert (min_timestamp_with_lookahead - min_timestamp_sans_lookahead).days >= max(
        cfg.preprocessing.pre_split.lookbehind_combination,
    ) - min_lookbehind - 1
