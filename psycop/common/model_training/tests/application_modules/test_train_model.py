"""Test that the model trains correctly."""


from pathlib import Path

import pytest

from psycop.common.model_training.application_modules.train_model.main import (
    post_wandb_setup_train_model,
    train_model,
)
from psycop.common.model_training.application_modules.wandb_handler import WandbHandler
from psycop.common.model_training.config_schemas.conf_utils import load_test_cfg_as_pydantic
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.common.model_training.training.model_specs import MODELS

INTEGRATION_TEST_FILE_NAME = "default_config.yaml"


@pytest.mark.parametrize("model_name", MODELS.keys())
def test_train_model(model_name: str):
    """Test main using a variety of model."""
    cfg: FullConfigSchema = load_test_cfg_as_pydantic(
        config_file_name=INTEGRATION_TEST_FILE_NAME, overrides=[f"model={model_name}"]
    )

    train_model(cfg)


def test_crossvalidation(muteable_test_config: FullConfigSchema):
    """Test crossvalidation."""
    cfg = muteable_test_config
    train_model(cfg)


def test_list_of_data_dirs(muteable_test_config: FullConfigSchema):
    """Test train model can resolve list of data dir paths."""
    cfg = muteable_test_config
    cfg.data.dir = [
        Path("psycop/common/model_training/tests/test_data/synth_splits_subsampled/"),
        Path("psycop/common/model_training/tests/test_data/synth_splits_subsampled/"),
    ]  # type: ignore
    train_model(cfg)


def test_train_val_predict(muteable_test_config: FullConfigSchema):
    """Test train without crossvalidation."""
    cfg = muteable_test_config
    cfg.data.splits_for_evaluation = ["test"]
    train_model(cfg)


def test_min_prediction_time_date(muteable_test_config: FullConfigSchema):
    """Test minimum prediction times correctly resolving the string."""
    cfg = muteable_test_config
    cfg.preprocessing.pre_split.min_prediction_time_date = "1972-01-01"
    train_model(cfg)


def test_feature_selection(muteable_test_config: FullConfigSchema):
    """Test feature selection."""
    cfg = muteable_test_config
    cfg.preprocessing.post_split.feature_selection.model_config["frozen"] = False
    cfg.preprocessing.post_split.feature_selection.name = "mutual_info_classif"
    cfg.preprocessing.post_split.feature_selection.params["percentile"] = 10  # type: ignore
    train_model(cfg)


def test_self_healing_nan_select_percentile(muteable_test_config: FullConfigSchema):
    """Test that train_model raises an exception when getting NaN as input and
    using select_percentile for feature selection, since that is undefined.

    Then check that adding the decorator suppresses the exception and
    returns 0.5 for Optuna search.
    """
    cfg = muteable_test_config
    cfg.preprocessing.post_split.imputation_method = None
    cfg.preprocessing.post_split.feature_selection.params["percentile"] = 10  # type: ignore
    cfg.preprocessing.post_split.feature_selection.name = "mutual_info_classif"

    # Train without the wrapper
    with pytest.raises(ValueError, match=r".*Input X contains NaN.*"):  # noqa
        WandbHandler(cfg=cfg).setup_wandb()
        post_wandb_setup_train_model.__wrapped__(cfg)  # type: ignore

    # Train with the wrapper
    wrapped_return_value = post_wandb_setup_train_model(cfg)
    assert wrapped_return_value == 0.5
