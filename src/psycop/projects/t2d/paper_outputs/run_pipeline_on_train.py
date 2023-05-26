"""A script for taking the current best model and running it on the test set."""

from pathlib import Path

from psycop.common.model_training.application_modules.train_model.main import (
    train_model,
)
from psycop.projects.t2d.paper_outputs.selected_runs import BEST_DEV_PIPELINE
from psycop.projects.t2d.utils.pipeline_objects import PipelineRun, RunGroup
from wasabi import Printer

msg = Printer(timestamp=True)


def _get_test_run_name(pipeline_to_train: PipelineRun) -> str:
    return f"{pipeline_to_train.name}-eval-on-test"


def _get_test_group_name(pipeline_to_train: PipelineRun) -> str:
    return f"{str(pipeline_to_train.group.name)}-eval-on-test"


def _get_test_group_path(pipeline_to_train: PipelineRun) -> Path:
    return Path(
        pipeline_to_train.group.group_dir.parent
        / _get_test_group_name(pipeline_to_train=pipeline_to_train),
    )


def _get_test_pipeline_dir(pipeline_to_train: PipelineRun) -> Path:
    """Get the path to the directory where the pipeline is evaluated on the test set."""
    return _get_test_group_path(
        pipeline_to_train=pipeline_to_train,
    ) / _get_test_run_name(pipeline_to_train=pipeline_to_train)


def _train_pipeline_on_test(pipeline_to_train: PipelineRun):
    cfg = pipeline_to_train.inputs.cfg
    cfg.project.wandb.Config.allow_mutation = True
    cfg.data.Config.allow_mutation = True
    cfg.data.splits_for_evaluation = ["test"]

    override_dir = _get_test_pipeline_dir(pipeline_to_train=pipeline_to_train)
    msg.info(f"Evaluating to {override_dir}")

    train_model(
        cfg=cfg,
        override_output_dir=override_dir,
    )


def get_test_pipeline_run(pipeline_to_train: PipelineRun) -> PipelineRun:
    # Check if the pipeline has already been trained on the test set
    # If so, return the existing run
    pipeline_has_been_evaluated_on_test = _get_test_pipeline_dir(
        pipeline_to_train=pipeline_to_train,
    ).exists()

    if not pipeline_has_been_evaluated_on_test:
        msg.info(
            f"{pipeline_to_train.group.name}/{pipeline_to_train.name} has not been evaluated on train, evaluating",
        )
        _train_pipeline_on_test(pipeline_to_train=pipeline_to_train)
    else:
        msg.good(
            f"{pipeline_to_train.group.name}/{pipeline_to_train.name} has already been evaluated on train, returning",
        )

    return PipelineRun(
        group=RunGroup(name=_get_test_group_name(pipeline_to_train)),
        name=_get_test_run_name(pipeline_to_train),
        pos_rate=0.03,
    )


if __name__ == "__main__":
    eval_run = get_test_pipeline_run(pipeline_to_train=BEST_DEV_PIPELINE)
