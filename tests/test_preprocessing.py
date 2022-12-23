"""Test custom preprocessing steps."""
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
