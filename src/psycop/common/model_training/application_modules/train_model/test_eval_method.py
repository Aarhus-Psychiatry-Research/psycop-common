from psycop.common.model_training.application_modules.train_model.main import (
    train_model,
)
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema


def test_crossvalidation(muteable_test_config: FullConfigSchema):
    """Test crossvalidation."""
    cfg = muteable_test_config
    train_model(cfg)


def test_train_val_predict(muteable_test_config: FullConfigSchema):
    """Test train without crossvalidation."""
    cfg = muteable_test_config
    cfg.data.splits_for_evaluation = ["test"]
    train_model(cfg)
