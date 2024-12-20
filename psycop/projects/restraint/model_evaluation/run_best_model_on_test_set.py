"""A script for taking the current best model and running it on the test set."""

from psycop.common.model_training.application_modules.train_model.main import train_model
from psycop.projects.restraint.model_evaluation.config import BEST_DEV_RUN, TEXT_BEST_DEV_RUN

if __name__ == "__main__":
    # Run best model trained on structured features only
    run_to_train_from = BEST_DEV_RUN

    cfg = run_to_train_from.cfg
    cfg.project.wandb.model_config["frozen"] = False
    cfg.project.wandb.group = f"{run_to_train_from.group.name}-eval-on-test"
    cfg.data.model_config["frozen"] = False
    cfg.data.splits_for_evaluation = ["val"]

    train_model(cfg=cfg)

    # Run best model trained on structured features and text features
    text_run_to_train_from = TEXT_BEST_DEV_RUN

    text_cfg = text_run_to_train_from.cfg
    text_cfg.project.wandb.model_config["frozen"] = False
    text_cfg.project.wandb.group = f"{text_run_to_train_from.group.name}-eval-on-test"
    text_cfg.data.model_config["frozen"] = False
    text_cfg.data.splits_for_evaluation = ["val"]

    train_model(cfg=text_cfg)
