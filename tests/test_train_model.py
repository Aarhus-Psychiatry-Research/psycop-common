import pytest
from hydra import compose, initialize

from psycopt2d.models import MODELS
from psycopt2d.train_model import main


@pytest.mark.parametrize("model_name", MODELS.keys())
def test_main(model_name):
    """test main using a variety of model."""
    with initialize(version_base=None, config_path="../src/psycopt2d/config/"):
        cfg = compose(
            config_name="test_config.yaml",
            overrides=[f"+model={model_name}"],
        )
        main(cfg)
