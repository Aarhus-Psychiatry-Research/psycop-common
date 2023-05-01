"""Testing of config schemas."""

import pytest
from hydra import compose, initialize
from psycop_model_training.config_schemas.conf_utils import (
    convert_omegaconf_to_pydantic_object,
)
from psycop_model_training.utils.utils import PROJECT_ROOT

CONFIG_DIR_PATH_TEST_ABS = PROJECT_ROOT / "tests" / "config"
CONFIG_DIR_PATH_TEST_REL = "../tests/config"

CONFIG_DIR_PATH_APP_ABS = PROJECT_ROOT / "application" / "config"
CONFIG_DIR_PATH_APP_REL = "../application/config"


def get_test_config_file_names() -> list[str]:
    """Get all config file names."""
    config_file_paths = list(CONFIG_DIR_PATH_TEST_ABS.glob("*.yaml"))

    return [f"{path.stem}.yaml" for path in config_file_paths]


@pytest.mark.parametrize("config_file_name", get_test_config_file_names())
def test_test_configs(config_file_name: str):
    """Test that all configs load correctly."""
    with initialize(version_base=None, config_path=CONFIG_DIR_PATH_TEST_REL):
        cfg = compose(
            config_name=config_file_name,
        )

    cfg = convert_omegaconf_to_pydantic_object(conf=cfg)


def get_app_config_file_names() -> list[str]:
    """Get all config file names."""
    config_file_paths = list(CONFIG_DIR_PATH_TEST_ABS.glob("*.yaml"))

    return [f"{path.stem}.yaml" for path in config_file_paths]
