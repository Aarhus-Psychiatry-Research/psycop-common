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
from psycop_model_training.data_loader.data_loader import DataLoader


@wandb_alert_on_exception
def main():
    """Main."""
    config_file_name = "default_config.yaml"

    cfg, wandb_group = setup(config_file_name=config_file_name)

    # Load dataset without dropping any rows for inferring
    # which look distances to grid search over
    train_df = DataLoader(cfg=cfg).load_dataset_from_dir(split_names="train")

    trainer_specs = SearchSpaceInferrer(
        cfg=cfg,
        train_df=train_df,
        model_names=["xgboost", "logistic_regression"],
    ).get_trainer_specs()

    spawn_trainers(
        cfg=cfg,
        config_file_name=config_file_name,
        wandb_prefix=wandb_group,
        trainer_specs=trainer_specs,
    )


if __name__ == "__main__":
    main()
