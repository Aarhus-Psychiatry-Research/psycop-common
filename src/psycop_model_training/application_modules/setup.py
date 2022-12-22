import wandb
from random_word import RandomWords

from psycop_model_training.utils.config_schemas.conf_utils import (
    load_app_cfg_as_pydantic,
)
from psycop_model_training.utils.config_schemas.full_config import FullConfigSchema


def setup(config_file_name: str) -> None:
    cfg = load_app_cfg_as_pydantic(config_file_name=config_file_name)
    wandb_group = setup_wandb(cfg=cfg)

    return cfg, wandb_group


def setup_wandb(cfg: FullConfigSchema) -> str:
    """Start a wandb group for this set of models."""
    random_word = RandomWords()
    wandb_group = f"{random_word.get_random_word()}-{random_word.get_random_word()}"

    wandb.init(
        project=f"{cfg.project.name}-baseline-model-training",
        mode=cfg.project.wandb.mode,
        group=wandb_group,
        entity=cfg.project.wandb.entity,
        name="process_manager",
    )

    return wandb_group
