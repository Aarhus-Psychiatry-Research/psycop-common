import pathlib
from collections.abc import Sequence
from typing import Any, Callable

import polars as pl

import psycop as ps
from psycop.common.cohort_definition import FilteredPredictionTimeBundle, StepDelta
from psycop.common.global_utils.cache import shared_cache
from psycop.common.global_utils.mlflow.mlflow_data_extraction import (
    MlflowClientWrapper,
    PsycopMlflowRun,
    filled_cfg_from_run,
)
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.loggers.terminal_logger import TerminalLogger
from psycop.common.model_training_v2.trainer.preprocessing.pipeline import (
    BaselinePreprocessingPipeline,
    PreprocessingPipeline,
)
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    CVDCohortDefiner,
    cvd_pred_filtering,
)
from psycop.projects.cvd.model_training.populate_cvd_registry import populate_with_cvd_registry


@shared_cache.cache
def _apply_preprocessing_pipeline(
    pre_steps: Sequence[StepDelta] | None,
    flattened_data: pl.LazyFrame,
    preprocessing_pipeline: PreprocessingPipeline,
) -> tuple[Sequence[StepDelta], pl.LazyFrame]:
    steps = list(pre_steps) if pre_steps is not None else []

    prior = len(flattened_data.collect())
    print(f"Rows in CohortDefiner: {prior}")

    i = len(steps)
    for step in preprocessing_pipeline.steps:
        flattened_data = step.apply(flattened_data)
        post_step_rows = len(flattened_data.collect())

        diff = prior - post_step_rows
        print(f"Rows dropped by {step.__class__.__name__}: {diff}")
        prior = post_step_rows
        steps.append(
            StepDelta(
                step_index=i,
                step_name=step.__class__.__name__,
                n_prediction_times_after=post_step_rows,
                n_prediction_times_before=prior,
                n_ids_after=0,
                n_ids_before=0,
            )
        )
        i += 1

    return steps, flattened_data


def filtering_flowchart_facade(
    prediction_time_bundle: FilteredPredictionTimeBundle,
    run: PsycopMlflowRun,
    output_dir: pathlib.Path,
):
    cfg = run.get_config()

    filled = filled_cfg_from_run(
        run, populate_registry_fns=[populate_baseline_registry, populate_with_cvd_registry]
    )

    pipeline: BaselinePreprocessingPipeline = filled["trainer"].preprocessing_pipeline
    pipeline._logger = TerminalLogger()  # type: ignore
    pipeline.steps[0].splits_to_keep = ["train", "val", "test"]  # type: ignore # Do not filter by region

    flattened_data = pl.scan_parquet(cfg["trainer"]["training_data"]["paths"][0]).lazy()

    step_deltas, flattened_data = _apply_preprocessing_pipeline(
        pre_steps=prediction_time_bundle.filter_steps,
        flattened_data=flattened_data,
        preprocessing_pipeline=pipeline,
    )

    def _stepdelta_line(prior: StepDelta, cur: StepDelta) -> str:
        return f"{cur.step_name}: Dropped {prior.n_prediction_times_before - cur.n_prediction_times_after:,}\n\t Remaining: {cur.n_prediction_times_after:,}"

    lines = [_stepdelta_line(prior, cur) for prior, cur in zip(step_deltas, step_deltas[1:])]

    outcome_matcher = pl.col(cfg["trainer"]["outcome_col_name"]) == pl.lit(1)
    lines.append(f"With outcome: {len(flattened_data.filter(outcome_matcher).collect()):,}")
    lines.append(
        f"Without outcome: {len(flattened_data.filter(outcome_matcher.not_()).collect()):,}"
    )

    # Output to a file
    (output_dir / "filtering_flowchart.csv").write_text("\n".join(lines))


if __name__ == "__main__":
    filtering_flowchart_facade(
        prediction_time_bundle=cvd_pred_filtering(),
        run=MlflowClientWrapper().get_run(experiment_name="CVD", run_name="CVD layer 1, base"),
        output_dir=pathlib.Path(),
    )
