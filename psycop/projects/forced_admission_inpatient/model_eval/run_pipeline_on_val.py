"""A script for taking the current best model and running it on the test set."""

from pathlib import Path

from config import MODEL_NAME
from wasabi import Printer

from psycop.common.model_training.application_modules.train_model.main import (
    train_model,
)
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    PipelineRun,
    RunGroup,
)

msg = Printer(timestamp=True)  # type: ignore


def _get_test_run_name(pipeline_to_train: PipelineRun) -> str:
    return f"{pipeline_to_train.name}-eval-on-test"


def _get_test_group_name(pipeline_to_train: PipelineRun) -> str:
    return f"{pipeline_to_train.group.group_name}-eval-on-test"


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
    cfg.data.splits_for_evaluation = ["val"]

    override_dir = _get_test_pipeline_dir(pipeline_to_train=pipeline_to_train)
    msg.info(f"Evaluating to {override_dir}")

    train_model(
        cfg=cfg,
        override_output_dir=override_dir,
    )


def test_pipeline(
    pipeline_to_test: PipelineRun,
) -> PipelineRun:
    # Check if the pipeline has already been trained on the test set
    # If so, return the existing run
    pipeline_has_been_evaluated_on_test = _get_test_pipeline_dir(
        pipeline_to_train=pipeline_to_test,
    ).exists()

    if not pipeline_has_been_evaluated_on_test:
        msg.info(
            f"{pipeline_to_test.group.group_name}/{pipeline_to_test.name} has not been evaluated, training",
        )
        _train_pipeline_on_test(pipeline_to_train=pipeline_to_test)
    else:
        msg.good(
            f"{pipeline_to_test.group.group_name}/{pipeline_to_test.name} has been evaluated, loading",
        )

    return PipelineRun(
        group=RunGroup(
            model_name=MODEL_NAME,
            group_name=_get_test_group_name(pipeline_to_test),
        ),
        name=_get_test_run_name(pipeline_to_test),
        pos_rate=0.03,
    )


if __name__ == "__main__":
    from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
        BEST_DEV_PIPELINE,
    )

    eval_run = test_pipeline(pipeline_to_test=BEST_DEV_PIPELINE)
