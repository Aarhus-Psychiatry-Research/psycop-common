"""Test custom preprocessing steps."""
from psycop_model_training.config.schemas import FullConfigSchema
from psycop_model_training.load import load_train_from_cfg
from psycop_model_training.preprocessing.post_split.create_pipeline import (
    create_preprocessing_pipeline,
)


def test_drop_datetime_predictor_columns(
    muteable_test_config: FullConfigSchema,
):
    """Test that columns are dropped if their lookbehind is not in the
    specified lookbehind combination list."""
    cfg = muteable_test_config

    cfg.preprocessing.drop_datetime_predictor_columns = True
    cfg.preprocessing.imputation_method = None
    cfg.preprocessing.feature_selection.name = None
    cfg.preprocessing.scaling = None
    cfg.data.pred_prefix = "timestamp"

    pipe = create_preprocessing_pipeline(cfg=cfg)
    train_df = load_train_from_cfg(cfg=cfg)
    train_df = pipe.transform(X=train_df)

    assert len([x for x in train_df.columns if "timestamp" in x]) == 0
