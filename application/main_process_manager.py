"""Example script to train multiple models and subsequently log the results to
wandb.

Usage:
- Replace the HYDRA_ARGS string with the desired arguments for `train_model.py`
- Run this script from project root with `python src/psycop_model_training/train_and_log_models.py
"""

from psycopmlutils.wandb.wandb_try_except_decorator import wandb_alert_on_exception

from psycop_model_training.application_modules.get_search_space import (
    SearchSpaceInferrer,
)
from psycop_model_training.application_modules.process_manager_setup import setup
from psycop_model_training.application_modules.trainer_spawner import spawn_trainers
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.data_loader.data_loader import DataLoader


@wandb_alert_on_exception
def main(
    cfg: FullConfigSchema,
    wandb_group: str,
):
    """Main."""
    # Load dataset without dropping any rows for inferring
    # which look distances to grid search over
    train_df = DataLoader(cfg=cfg).load_dataset_from_dir(split_names="train")

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
    )


if __name__ == "__main__":
    CONFIG_FILE_NAME = "default_config.yaml"

    # Must run cfg before main to ensure that wandb is initialized
    # before adding wandb_alert_on_exception decorator
    cfg, wandb_group = setup(config_file_name=CONFIG_FILE_NAME)

    main(cfg=cfg, wandb_group=wandb_group)
