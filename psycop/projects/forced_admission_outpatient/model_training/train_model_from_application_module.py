"""Script using the train_model module to train a model.

Required to allow the trainer_spawner to point towards a python script
file, rather than an installed module.
"""
import faulthandler
import sys
from pathlib import Path
from typing import Union

import hydra
from omegaconf import DictConfig

from psycop.common.model_training.application_modules.train_model.main import train_model
from psycop.common.model_training.config_schemas.conf_utils import (
    convert_omegaconf_to_pydantic_object,
)
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "model_training" / "config"


@hydra.main(config_path=str(CONFIG_PATH), config_name="default_config", version_base="1.2")
def main(cfg: Union[DictConfig, FullConfigSchema]) -> float:
    """Main."""
    if not isinstance(cfg, FullConfigSchema):
        cfg = convert_omegaconf_to_pydantic_object(cfg)

    if sys.platform == "win32":
        (PROJECT_ROOT / "wandb" / "debug-cli.onerm").mkdir(exist_ok=True, parents=True)

    return train_model(cfg=cfg)


if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    main()
