"""A script for taking the current best model and running it on the test set."""

from pathlib import Path

from wasabi import Printer

from psycop.common.model_training.application_modules.train_model.main import train_model
from psycop.projects.t2d.utils.pipeline_objects import T2DPipelineRun

msg = Printer(timestamp=True)


def _get_test_run_name(pipeline_to_train: T2DPipelineRun) -> str:
    return f"{pipeline_to_train.name}-eval-on-test"


def _get_test_group_name(pipeline_to_train: T2DPipelineRun) -> str:
    return f"{pipeline_to_train.group.name!s}-eval-on-test"


def _get_test_group_path(pipeline_to_train: T2DPipelineRun) -> Path:
    return Path(
        pipeline_to_train.group.group_dir.parent
        / _get_test_group_name(pipeline_to_train=pipeline_to_train)
    )


def _get_test_pipeline_dir(pipeline_to_train: T2DPipelineRun) -> Path:
    """Get the path to the directory where the pipeline is evaluated on the test set."""
    return _get_test_group_path(pipeline_to_train=pipeline_to_train) / _get_test_run_name(
        pipeline_to_train=pipeline_to_train
    )


def _train_pipeline_on_test(pipeline_to_train: T2DPipelineRun):
    cfg = pipeline_to_train.inputs.cfg
    cfg.project.wandb.model_config["frozen"] = False
    cfg.data.model_config["frozen"] = False
    cfg.data.splits_for_evaluation = ["test"]

    override_dir = _get_test_pipeline_dir(pipeline_to_train=pipeline_to_train)
    msg.info(f"Evaluating to {override_dir}")

    train_model(cfg=cfg, override_output_dir=override_dir)


def test_pipeline(pipeline_to_test: T2DPipelineRun) -> T2DPipelineRun:
    # Check if the pipeline has already been trained on the test set
    # If so, return the existing run
    pipeline_has_been_evaluated_on_test = _get_test_pipeline_dir(
        pipeline_to_train=pipeline_to_test
    ).exists()

    if not pipeline_has_been_evaluated_on_test:
        msg.info(
            f"{pipeline_to_test.group.name}/{pipeline_to_test.name} has not been evaluated, training"
        )
        _train_pipeline_on_test(pipeline_to_train=pipeline_to_test)
    else:
        msg.good(
            f"{pipeline_to_test.group.name}/{pipeline_to_test.name} has been evaluated, loading"
        )

    return pipeline_to_test


if __name__ == "__main__":
    from psycop.projects.t2d.paper_outputs.selected_runs import get_best_dev_pipeline

    eval_run = test_pipeline(pipeline_to_test=get_best_dev_pipeline())
