"""Example script to train multiple models and subsequently log the results to
wandb.
"""

from pathlib import Path

from psycop.common.model_training.application_modules.get_search_space import (
    SearchSpaceInferrer,
)
from psycop.common.model_training.application_modules.process_manager_setup import setup
from psycop.common.model_training.application_modules.trainer_spawner import (
    spawn_trainers,
)
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.common.model_training.data_loader.data_loader import DataLoader
from psycop_ml_utils.wandb.wandb_try_except_decorator import wandb_alert_on_exception


@wandb_alert_on_exception
def main(
    cfg: FullConfigSchema,
    wandb_group: str,
):
    """Main."""
    # Load dataset without dropping any rows for inferring
    # which look distances to grid search over
    train_df = DataLoader(data_cfg=cfg.data).load_dataset_from_dir(split_names="train")

    trainer_specs = SearchSpaceInferrer(
        cfg=cfg,
        train_df=train_df,
        model_names=["xgboost", "logistic-regression"],
    ).get_trainer_specs()

    spawn_trainers(
        cfg=cfg,
        config_file_name=CONFIG_FILE_NAME,
        wandb_prefix=wandb_group,
        trainer_specs=trainer_specs,
        train_single_model_file_path=Path(
            "src/care_ml/model_training/application/train_model_from_application_module.py",
        ),
    )


if __name__ == "__main__":
    CONFIG_FILE_NAME = "default_config.yaml"

    # Must run cfg before main to ensure that wandb is initialized
    # before adding wandb_alert_on_exception decorator
    cfg, wandb_group = setup(
        config_file_name=CONFIG_FILE_NAME,
        application_config_dir_relative_path="../../../../../../care_ml/model_training/application/config/",
    )

    main(cfg=cfg, wandb_group=wandb_group)
