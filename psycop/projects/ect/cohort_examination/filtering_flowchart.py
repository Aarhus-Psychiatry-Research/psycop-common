from pathlib import Path
from collections.abc import Sequence
from tempfile import mkdtemp

import polars as pl

from confection import Config
from psycop.common.cohort_definition import FilteredPredictionTimeBundle, StepDelta
from psycop.common.global_utils.cache import shared_cache
from psycop.common.global_utils.mlflow.mlflow_data_extraction import (
    MlflowClientWrapper,
    PsycopMlflowRun,
    filled_cfg_from_run,
)
from psycop.common.model_training_v2.config.config_utils import PsycopConfig, resolve_and_fill_config
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.loggers.terminal_logger import TerminalLogger
from psycop.common.model_training_v2.trainer.preprocessing.pipeline import (
    BaselinePreprocessingPipeline,
    PreprocessingPipeline,
)
from psycop.projects.ect.feature_generation.cohort_definition.ect_cohort_definition import (
    ect_pred_filtering,
)
from psycop.projects.restraint.evaluation.utils import read_eval_df_from_disk


@shared_cache().cache
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
    cfg: PsycopConfig,
    output_dir: Path,
):

    tmp_cfg = Path(mkdtemp()) / "tmp.cfg"
    cfg.to_disk(tmp_cfg)
    filled = resolve_and_fill_config(tmp_cfg, fill_cfg_with_defaults=True)

    pipeline: BaselinePreprocessingPipeline = filled["trainer"].training_preprocessing_pipeline
    pipeline._logger = TerminalLogger()  # type: ignore
    pipeline.steps[0].splits_to_keep = ["train", "val", "test"]  # type: ignore # Do not filter by split

    flattened_data = pl.scan_parquet(cfg["trainer"]["training_data"]["paths"][0]).lazy()

    step_deltas, flattened_data = _apply_preprocessing_pipeline(
        pre_steps=prediction_time_bundle.filter_steps,
        flattened_data=flattened_data,
        preprocessing_pipeline=pipeline,
    )

    def _stepdelta_line(prior: StepDelta, cur: StepDelta) -> str:
        return f"{cur.step_name}: Dropped {prior.n_prediction_times_before - cur.n_prediction_times_after:,}\n\t Remaining: {cur.n_prediction_times_after:,}"

    lines = [_stepdelta_line(prior, cur) for prior, cur in zip(step_deltas, step_deltas[1:])]

    outcome_matcher = pl.col(cfg["trainer"]["training_outcome_col_name"]) == pl.lit(1)
    lines.append(f"With outcome: {len(flattened_data.filter(outcome_matcher).collect()):,}")
    lines.append(
        f"Without outcome: {len(flattened_data.filter(outcome_matcher.not_()).collect()):,}"
    )
    
    # Output to a file
    (output_dir / "filtering_flowchart.csv").write_text("\n".join(lines))



if __name__ == "__main_n":
        experiment="ECT-hparam-structured_only-xgboost-no-lookbehind-filter"
        experiment_path = f"E:/shared_resources/ect/eval_runs/{experiment}_best_run_evaluated_on_test"
        experiment_df = read_eval_df_from_disk(experiment_path)
        experiment_cfg = PsycopConfig(Config().from_disk
                                      (path=Path(experiment_path) / 'config.cfg'))
        
        save_dir = Path(experiment_path + "/figures") 
        save_dir.mkdir(parents=True, exist_ok=True)

        filtering_flowchart_facade(
            prediction_time_bundle=ect_pred_filtering(), cfg=experiment_cfg, output_dir=save_dir
        )