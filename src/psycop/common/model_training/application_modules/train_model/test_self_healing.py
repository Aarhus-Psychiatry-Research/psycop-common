import pytest
from psycop.common.model_training.application_modules.train_model.main import (
    post_wandb_setup_train_model,
)
from psycop.common.model_training.application_modules.wandb_handler import WandbHandler
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema


def test_self_healing_nan_select_percentile(muteable_test_config: FullConfigSchema):
    """Test that train_model raises an exception when getting NaN as input and
    using select_percentile for feature selection, since that is undefined.
    Then check that adding the decorator suppresses the exception and
    returns 0.5 for Optuna search.
    """
    cfg = muteable_test_config
    cfg.preprocessing.post_split.imputation_method = None
    cfg.preprocessing.post_split.feature_selection.params[  # type: ignore
        "percentile"
    ] = 10
    cfg.preprocessing.post_split.feature_selection.name = "mutual_info_classif"

    # Train without the wrapper
    with pytest.raises(ValueError, match=r".*Input X contains NaN.*"):  # noqa
        WandbHandler(cfg=cfg).setup_wandb()
        post_wandb_setup_train_model.__wrapped__(cfg)

    # Train with the wrapper
    wrapped_return_value = post_wandb_setup_train_model(cfg)
    assert wrapped_return_value == 0.5
