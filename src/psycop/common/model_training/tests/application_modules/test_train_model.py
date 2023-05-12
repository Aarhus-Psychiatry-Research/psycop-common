"""Test that the model trains correctly."""


import pytest
from psycop.common.model_training.application_modules.train_model.main import (
    train_model,
)
from psycop.common.model_training.config_schemas.conf_utils import (
    load_test_cfg_as_pydantic,
)
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.common.model_training.training.model_specs import MODELS

INTEGRATION_TEST_FILE_NAME = "default_config.yaml"


@pytest.mark.parametrize("model_name", MODELS.keys())
def test_train_model(model_name: str):
    """Test main using a variety of model."""
    cfg: FullConfigSchema = load_test_cfg_as_pydantic(
        config_file_name=INTEGRATION_TEST_FILE_NAME,
        overrides=[f"model={model_name}"],
    )

    train_model(cfg)
