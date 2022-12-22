"""Test that the model trains correctly."""


import pytest

from application.train_model import main
from psycop_model_training.training.model_specs import MODELS
from psycop_model_training.utils.config_schemas.conf_utils import (
    load_test_cfg_as_omegaconf,
)
from psycop_model_training.utils.config_schemas.full_config import FullConfigSchema

INTEGRATION_TEST_FILE_NAME = "default_config.yaml"


@pytest.mark.parametrize("model_name", MODELS.keys())
def test_main(model_name: str):
    """Test main using a variety of model."""

    cfg: FullConfigSchema = load_test_cfg_as_omegaconf(
        config_file_name=INTEGRATION_TEST_FILE_NAME,
        overrides=[f"model={model_name}"],
    )

    main(cfg)


@pytest.mark.pre_push_test
def test_integration_test(muteable_test_config: FullConfigSchema):
    """Test main using the logistic model.

    Used for quickly testing functions before a push.
    """
    cfg = muteable_test_config
    cfg.eval.force = True
    main(cfg)


def test_crossvalidation(muteable_test_config: FullConfigSchema):
    """Test crossvalidation."""
    cfg = muteable_test_config
    cfg.train.n_splits = 2
    main(cfg)


def test_min_prediction_time_date(muteable_test_config: FullConfigSchema):
    """Test minimum prediction times correctly resolving the string."""
    cfg = muteable_test_config
    cfg.preprocessing.pre_split.min_prediction_time_date = "1972-01-01"
    main(cfg)


def test_feature_selection(muteable_test_config: FullConfigSchema):
    """Test feature selection."""
    cfg = muteable_test_config
    cfg.preprocessing.post_split.feature_selection.name = "mutual_info_classif"
    cfg.preprocessing.post_split.feature_selection.params["percentile"] = 10
    main(cfg)
