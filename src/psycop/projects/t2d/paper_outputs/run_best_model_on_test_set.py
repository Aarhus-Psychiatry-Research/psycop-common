"""A script for taking the current best model and running it on the test set."""

from psycop.common.model_training.application_modules.train_model.main import (
    train_model,
)
from psycop.projects.t2d.paper_outputs.config import BEST_DEV_RUN
from psycop.projects.t2d.utils.best_runs import PipelineRun


def run_pipeline_on_train(pipeline_to_train: PipelineRun) -> PipelineRun:
    # Check if the pipeline has already been trained on the test set
    # If so, return the existing run
    test_pipeline_group_name = f"{pipeline_to_train.group.name}-eval-on-test"

    pipeline_has_been_evaluated_on_test = path_to_test_dir.exist()

    if pipeline_has_been_evaluated_on_test:
        return PipelineRun(
            group=RunGroup(name=test_pipeline_group_name),
            name=
        )

    cfg = pipeline_to_train.cfg
    cfg.project.wandb.Config.allow_mutation = True
    cfg.project.wandb.group = test_pipeline_group_name
    cfg.data.Config.allow_mutation = True
    cfg.data.splits_for_evaluation = ["test"]

    train_model(cfg=cfg)
    return None


if __name__ == "__main__":
    run_pipeline_on_train(pipeline_to_train=BEST_DEV_RUN)
