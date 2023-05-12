"""A script for taking the current best model and running it on the test set."""

from psycop.model_training.application_modules.train_model.main import train_model
from psycop.projects.t2d.paper_outputs.config import BEST_DEV_RUN

if __name__ == "__main__":
    run_to_train_from = BEST_DEV_RUN

    cfg = run_to_train_from.cfg
    cfg.project.wandb.Config.allow_mutation = True
    cfg.project.wandb.group = f"{run_to_train_from.group.name}-eval-on-test"
    cfg.data.Config.allow_mutation = True
    cfg.data.splits_for_evaluation = ["test"]

    train_model(cfg=cfg)
