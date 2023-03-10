import wandb
from psycop_model_training.config_schemas.conf_utils import load_app_cfg_as_pydantic
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from random_word import RandomWords


def create_random_wandb_group_name() -> str:
    """Create a random wandb group name."""
    random_word = RandomWords()
    wandb_group = f"{random_word.get_random_word()}-{random_word.get_random_word()}"
    return wandb_group


def setup_wandb(cfg: FullConfigSchema) -> str:
    """Start a wandb group for this set of models."""
    wandb_group = create_random_wandb_group_name()

    wandb.init(
        project=f"{cfg.project.name}-baseline-model-training",
        mode=cfg.project.wandb.mode,
        group=wandb_group,
        entity=cfg.project.wandb.entity,
        name="process_manager",
    )

    return wandb_group


def setup(
    config_file_name: str,
    application_config_dir_relative_path: str,
) -> tuple[FullConfigSchema, str]:
    """Setup the requirements to run the model training pipeline."""
    cfg = load_app_cfg_as_pydantic(
        config_file_name=config_file_name,
        config_dir_path_rel=application_config_dir_relative_path,
    )
    wandb_group = setup_wandb(cfg=cfg)

    return cfg, wandb_group
