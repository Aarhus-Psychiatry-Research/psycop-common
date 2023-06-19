"""Script using the train_model module to train a model.

Required to allow the trainer_spawner to point towards a python script
file, rather than an installed module.
"""
from pathlib import Path

import hydra
from omegaconf import DictConfig
from psycop.common.model_training.application_modules.train_model.main import (
    train_model,
)
from psycop.common.model_training.config_schemas.conf_utils import (
    convert_omegaconf_to_pydantic_object,
)
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "application" / "config"
WANDB_LOG_PATH = PROJECT_ROOT / "wandb" / "debug-cli.onerm"


@hydra.main(
    config_path=str(CONFIG_PATH),
    config_name="default_config",
    version_base="1.2",
)
def main(cfg: DictConfig) -> float:
    """Main."""
    if not Path.exists(WANDB_LOG_PATH):
        WANDB_LOG_PATH.mkdir(parents=True, exist_ok=True)

    if not isinstance(cfg, FullConfigSchema):
        cfg = convert_omegaconf_to_pydantic_object(cfg)  # type: ignore

    return train_model(cfg=cfg)  # type: ignore


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
