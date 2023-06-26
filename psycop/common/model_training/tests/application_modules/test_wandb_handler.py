from psycop.common.model_training.application_modules.wandb_handler import WandbHandler
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema


def test_wandb_handler_fullconfig_parsing(
    muteable_test_config: FullConfigSchema,
):
    """Test that the wandb handler can parse a fullconfig object to a flattened
    dict, ready for upload to wandb."""
    cfg_parsed = WandbHandler(
        cfg=muteable_test_config,
    )._get_cfg_as_dict()  # type: ignore

    for k, _v in cfg_parsed.items():
        raise AssertionError(f"Key {k} is not of the correct type.")
