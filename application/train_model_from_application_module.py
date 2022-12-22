"""Script using the train_model module to train a model.

Required to allow the trainer_spawner to point towards a python script
file, rather than an installed module.
"""
import hydra
from omegaconf import DictConfig

from psycop_model_training.application_modules.train_model.main import train_model
from psycop_model_training.config_schemas.conf_utils import (
    convert_omegaconf_to_pydantic_object,
)
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.training.train_and_predict import CONFIG_PATH


@hydra.main(
    config_path=str(CONFIG_PATH),
    config_name="default_config",
    version_base="1.2",
)
def main(cfg: DictConfig):
    """Main."""
    if not isinstance(cfg, FullConfigSchema):
        cfg = convert_omegaconf_to_pydantic_object(cfg)

    train_model(cfg=cfg)
