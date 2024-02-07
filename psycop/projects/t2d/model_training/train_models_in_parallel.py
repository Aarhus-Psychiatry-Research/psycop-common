"""Example script to train multiple models and subsequently log the results to
wandb.
"""

from pathlib import Path
from typing import Optional

from psycop.common.model_training.application_modules.get_search_space import SearchSpaceInferrer
from psycop.common.model_training.application_modules.process_manager_setup import setup
from psycop.common.model_training.application_modules.trainer_spawner import spawn_trainers
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.common.model_training.data_loader.data_loader import DataLoader


def main(
    cfg: FullConfigSchema,
    wandb_group: str,
    config_file_name: str,
    dataset_override_path: Optional[Path] = None,
):
    """Main."""
    if dataset_override_path is not None:
        cfg.data.model_config["frozen"] = False
        cfg.data.dir = dataset_override_path

    # Load dataset without dropping any rows for inferring
    # which look distances to grid search over
    train_df = DataLoader(data_cfg=cfg.data).load_dataset_from_dir(split_names="val")

    trainer_specs = SearchSpaceInferrer(
        cfg=cfg, train_df=train_df, model_names=["xgboost", "logistic-regression"]
    ).get_trainer_specs()

    spawn_trainers(
        cfg=cfg,
        config_file_name=config_file_name,
        wandb_prefix=wandb_group,
        trainer_specs=trainer_specs,
        train_single_model_file_path=Path(
            "psycop/projects/t2d/model_training/train_model_from_application_module.py"
        ),
    )


def train_models_in_parallel(dataset_override_path: Optional[Path] = None):
    CONFIG_FILE_NAME = "default_config.yaml"

    # Must run cfg before main to ensure that wandb is initialized
    # before adding wandb_alert_on_exception decorator
    cfg, wandb_group = setup(
        config_file_name=CONFIG_FILE_NAME,
        application_config_dir_relative_path="../../../../psycop/projects/t2d/model_training/config/",
    )

    main(
        cfg=cfg,
        wandb_group=wandb_group,
        dataset_override_path=dataset_override_path,
        config_file_name=CONFIG_FILE_NAME,
    )


if __name__ == "__main__":
    train_models_in_parallel()
