"""A script for taking the current best model and running it on the test set."""

from pathlib import Path

from psycop.common.model_training.application_modules.train_model.main import (
    train_model,
)
from psycop.projects.t2d.paper_outputs.config import BEST_DEV_RUN
from psycop.projects.t2d.utils.best_runs import PipelineRun, RunGroup
from wasabi import Printer

msg = Printer(timestamp=True)


def get_test_run_name(pipeline_to_train: PipelineRun) -> str:
    return f"{pipeline_to_train.name}-eval-on-test"


def get_test_group_name(pipeline_to_train: PipelineRun) -> str:
    return f"{str(pipeline_to_train.group.name)}-eval-on-test"


def get_test_group_path(pipeline_to_train: PipelineRun) -> Path:
    return Path(
        pipeline_to_train.group.group_dir.parent
        / get_test_group_name(pipeline_to_train=pipeline_to_train)
    )


def get_test_pipeline_dir(pipeline_to_train: PipelineRun) -> Path:
    """Get the path to the directory where the pipeline is evaluated on the test set."""
    return get_test_group_path(pipeline_to_train=pipeline_to_train) / get_test_run_name(
        pipeline_to_train=pipeline_to_train
    )


def train_pipeline_on_test(pipeline_to_train: PipelineRun):
    cfg = pipeline_to_train.cfg
    cfg.project.wandb.Config.allow_mutation = True
    cfg.data.Config.allow_mutation = True
    cfg.data.splits_for_evaluation = ["test"]

    override_dir = get_test_pipeline_dir(pipeline_to_train=pipeline_to_train)
    msg.info(f"Evaluating to {override_dir}")

    train_model(
        cfg=cfg,
        override_output_dir=override_dir,
    )


def run_pipeline_on_train(pipeline_to_train: PipelineRun) -> PipelineRun:
    # Check if the pipeline has already been trained on the test set
    # If so, return the existing run
    pipeline_has_been_evaluated_on_test = get_test_pipeline_dir(
        pipeline_to_train=pipeline_to_train
    ).exists()

    if not pipeline_has_been_evaluated_on_test:
        msg.info(
            f"{pipeline_to_train.group.name}/{pipeline_to_train.name} has not been evaluated on train, evaluating"
        )
        train_pipeline_on_test(pipeline_to_train=pipeline_to_train)
    else:
        msg.good(
            f"{pipeline_to_train.group.name}/{pipeline_to_train.name} has already been evaluated on train, returning"
        )

    return PipelineRun(
        group=RunGroup(name=get_test_group_name(pipeline_to_train)),
        name=get_test_run_name(pipeline_to_train),
        pos_rate=0.03,
    )


if __name__ == "__main__":
    eval_run = run_pipeline_on_train(pipeline_to_train=BEST_DEV_RUN)
