from collections.abc import Sequence

import plotnine as pn
import polars as pl

# Set path to BaselineSchema for the run
## Load dataset with predictions after training
## Load validation dataset
## 1:1 join metadata cols to predictions
# Plot performance by
## Age
## Sex
## Calendar time
## Diagnosis type (scz or bp)
## Time to event
## Time from first visit
# Table of performance (sens, spec, ppv, f1) by threshold
## Confusion matrix at specified threshold
# Plot feature importance
from psycop.common.global_utils.mlflow.mlflow_data_extraction import (
    MlflowClientWrapper,
    PsycopMlflowRun,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.trainer.base_trainer import BaselineTrainer
from psycop.common.model_training_v2.trainer.cross_validator_trainer import CrossValidatorTrainer
from psycop.common.model_training_v2.trainer.split_trainer import SplitTrainer
from psycop.projects.scz_bp.evaluation.minimal_eval_dataset import (
    minimal_eval_dataset_from_mlflow_run,
)
from psycop.projects.scz_bp.evaluation.model_performance.performance.performance_by_time_to_event import (
    scz_bp_plot_sensitivity_by_time_to_event,
)
from psycop.projects.scz_bp.evaluation.model_performance.robustness.scz_bp_robustness_by_age import (
    scz_bp_auroc_by_age,
)
from psycop.projects.scz_bp.evaluation.model_performance.robustness.scz_bp_robustness_by_calendar_time import (
    scz_bp_auroc_by_quarter,
)
from psycop.projects.scz_bp.evaluation.model_performance.robustness.scz_bp_robustness_by_cyclic_time import (
    scz_bp_auroc_by_day_of_week,
    scz_bp_auroc_by_month_of_year,
)
from psycop.projects.scz_bp.evaluation.model_performance.robustness.scz_bp_robustness_by_sex import (
    scz_bp_auroc_by_sex,
)
from psycop.projects.scz_bp.evaluation.model_performance.robustness.scz_bp_robustness_by_time_from_first_visit import (
    scz_bp_auroc_by_time_from_first_contact,
)

populate_baseline_registry()


def scz_bp_df_to_eval_df(
    df: pl.DataFrame, y_hat_prop_col_name: str, y_col_name: str
) -> EvalDataset:
    return EvalDataset(
        ids=df["dw_ek_borger"].to_pandas(),
        pred_time_uuids=df["prediction_time_uuid"].to_pandas(),
        pred_timestamps=df["timestamp"].to_pandas(),
        outcome_timestamps=df["meta_time_of_diagnosis"].to_pandas(),
        y=df[y_col_name].to_pandas(),
        y_hat_probs=df[y_hat_prop_col_name].to_pandas(),
        age=df["pred_age_in_years"].to_pandas(),
        is_female=df["pred_sex_female_layer_1"].to_pandas(),
        custom_columns={
            "scz_or_bp": df["meta_scz_or_bp_indicator"].to_pandas(),
            "first_visit": df["meta_first_visit"].to_pandas(),
        },
    )


def _load_validation_data_from_schema(schema: BaselineSchema) -> pl.DataFrame:
    match schema.trainer:
        case CrossValidatorTrainer():
            return schema.trainer.training_data.load().collect()
        case SplitTrainer():
            return schema.trainer.validation_data.load().collect()
        case BaselineTrainer():
            raise TypeError("That's an ABC, mate")


def merge_pred_df_with_validation_df(
    pred_df: pl.DataFrame, validation_df: pl.DataFrame
) -> pl.DataFrame:
    validation_df = validation_df.select(pl.col("^meta.*$"), "timestamp", "prediction_time_uuid")
    return pred_df.join(validation_df, how="left", on="prediction_time_uuid")


def cohort_metadata_from_run(
    run: PsycopMlflowRun, cohort_metadata_cols: Sequence[pl.Expr]
) -> pl.DataFrame:
    cfg = run.get_config()
    schema = BaselineSchema(**BaselineRegistry.resolve(cfg))

    return _load_validation_data_from_schema(schema=schema).select(cohort_metadata_cols)


def full_eval(eval_ds: EvalDataset) -> list[pn.ggplot]:
    age = scz_bp_auroc_by_age(eval_ds=eval_ds)
    sex = scz_bp_auroc_by_sex(eval_ds=eval_ds)
    time_from_first_visit = scz_bp_auroc_by_time_from_first_contact(eval_ds=eval_ds)

    dow = scz_bp_auroc_by_day_of_week(eval_ds=eval_ds)
    month = scz_bp_auroc_by_month_of_year(eval_ds=eval_ds)
    quarter = scz_bp_auroc_by_quarter(eval_ds=eval_ds)

    sens_time_to_event = scz_bp_plot_sensitivity_by_time_to_event(eval_ds=eval_ds)
    return [age, sex, time_from_first_visit, dow, month, quarter, sens_time_to_event]


if __name__ == "__main__":
    experiment_name = "scz-bp_3_year_lookahead"
    best_run = MlflowClientWrapper().get_best_run_from_experiment(
        experiment_name=experiment_name, metric="all_oof_BinaryAUROC"
    )

    eval_ds = minimal_eval_dataset_from_mlflow_run(run=best_run)
    cohort_metadata = cohort_metadata_from_run(
        run=best_run,
        cohort_metadata_cols=[
            pl.col("prediction_time_uuid"),
            pl.col("dw_ek_borger"),
            pl.col("timestamp"),
            pl.col("^meta.*$"),
        ],
    )
    df = eval_ds.frame.join(cohort_metadata, how="left", on=eval_ds.pred_time_uuid_col_name)
