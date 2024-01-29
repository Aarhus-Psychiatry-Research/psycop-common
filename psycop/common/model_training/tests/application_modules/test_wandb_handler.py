from pathlib import Path

from psycop.common.model_training.application_modules.wandb_handler import WandbHandler
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema


def test_wandb_handler_fullconfig_parsing(muteable_test_config: FullConfigSchema):
    """Test that the wandb handler can parse a fullconfig object to a flattened
    dict, ready for upload to wandb."""
    cfg_parsed = WandbHandler(cfg=muteable_test_config)._get_cfg_as_dict()  # type: ignore

    for _k, v in cfg_parsed.items():
        if not isinstance(v, (str, int, float, Path, list)) and v is not None:
            raise AssertionError(f"Value {v} in config is not of the correct type.")
