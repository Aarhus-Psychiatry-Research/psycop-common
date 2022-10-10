"""Test that the model trains correctly."""

import pytest
from hydra import compose, initialize

from psycopt2d.models import MODELS
from psycopt2d.train_model import main


@pytest.mark.parametrize("model_name", MODELS.keys())
def test_main(model_name):
    """test main using a variety of model."""
    with initialize(version_base=None, config_path="../src/psycopt2d/config/"):
        cfg = compose(
            config_name="integration_testing.yaml",
            overrides=[f"+model={model_name}"],
        )
        main(cfg)


@pytest.mark.pre_push_test
def test_integration_test():
    """test main using the logistic model.

    Used for quickly testing functions before a push.
    """
    with initialize(version_base=None, config_path="../src/psycopt2d/config/"):
        cfg = compose(
            config_name="integration_testing.yaml",
            overrides=["+model=logistic-regression"],
        )
        main(cfg)


def test_crossvalidation():
    """Test crossvalidation."""
    with initialize(version_base=None, config_path="../src/psycopt2d/config/"):
        cfg = compose(
            config_name="integration_testing.yaml",
            overrides=["+model=logistic-regression", "+data.n_splits=2"],
        )
        main(cfg)


def test_min_prediction_time_date():
    """Test crossvalidation."""
    with initialize(version_base=None, config_path="../src/psycopt2d/config/"):
        cfg = compose(
            config_name="integration_testing.yaml",
            overrides=[
                "+model=logistic-regression",
                "+data.min_prediction_time_date=1972-01-01",
            ],
        )
        main(cfg)
