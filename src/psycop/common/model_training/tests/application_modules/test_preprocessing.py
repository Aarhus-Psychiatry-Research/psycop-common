from psycop.common.model_training.application_modules.train_model.main import (
    train_model,
)
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema


def test_feature_selection(muteable_test_config: FullConfigSchema):
    """Test feature selection."""
    cfg = muteable_test_config
    cfg.preprocessing.post_split.feature_selection.Config.allow_mutation = True
    cfg.preprocessing.post_split.feature_selection.name = "mutual_info_classif"
    cfg.preprocessing.post_split.feature_selection.params[  # type: ignore
        "percentile"
    ] = 10
    train_model(cfg)


def test_min_prediction_time_date(muteable_test_config: FullConfigSchema):
    """Test minimum prediction times correctly resolving the string."""
    cfg = muteable_test_config
    cfg.preprocessing.pre_split.min_prediction_time_date = "1972-01-01"
    train_model(cfg)
