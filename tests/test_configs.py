"""Testing of config schemas."""
from pathlib import Path

import pytest
from hydra import compose, initialize

from psycopt2d.utils.config_schemas import convert_omegaconf_to_pydantic_object
from psycopt2d.utils.utils import PROJECT_ROOT

CONFIG_DIR_PATH_ABS = PROJECT_ROOT / "src" / "psycopt2d" / "config"
CONFIG_DIR_PATH_REL = "../src/psycopt2d/config"


def get_config_file_names() -> list[str]:
    """Get all config file names."""
    config_file_paths: list[Path] = list(CONFIG_DIR_PATH_ABS.glob("*.yaml"))
    return [f"{path.stem}.yaml" for path in config_file_paths]


@pytest.mark.parametrize("config_file_name", get_config_file_names())
def test_configs(config_file_name):
    """Test that all configs load correctly."""
    with initialize(version_base=None, config_path=CONFIG_DIR_PATH_REL):
        cfg = compose(
            config_name=config_file_name,
        )

    cfg = convert_omegaconf_to_pydantic_object(conf=cfg)
