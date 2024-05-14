from collections.abc import Sequence
from pathlib import Path

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
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.baseline_schema import BaselineSchema
from psycop.common.model_training_v2.config.populate_registry import (
    populate_baseline_registry,
)
from psycop.common.model_training_v2.trainer.base_trainer import BaselineTrainer
from psycop.common.model_training_v2.trainer.cross_validator_trainer import (
    CrossValidatorTrainer,
)
from psycop.common.model_training_v2.trainer.split_trainer import SplitTrainer, SplitTrainerSeparatePreprocessing

from psycop.projects.scz_bp.evaluation.minimal_eval_dataset import minimal_eval_dataset_from_path
from psycop.projects.scz_bp.model_training.synthetic_trainer.synthetic_split_trainer import (
    SyntheticSplitTrainerSeparatePreprocessing,
)
from psycop.projects.scz_bp.model_training.populate_scz_bp_registry import (
    populate_scz_bp_registry,
)
from psycop.projects.scz_bp.model_training.synthetic_trainer.synthetic_cv_trainer import (
    SyntheticCrossValidatorTrainer,
)
from psycop.projects.scz_bp.model_training.synthetic_trainer.synthetic_split_trainer import (
    SyntheticSplitTrainerSeparatePreprocessing,
)

populate_baseline_registry()
populate_scz_bp_registry()

def scz_bp_df_to_eval_df(df: pl.DataFrame) -> EvalDataset:
    return EvalDataset(
        ids=df["dw_ek_borger"].to_pandas(),
        pred_time_uuids=df["pred_time_uuid"].to_pandas(),
        pred_timestamps=df["timestamp"].to_pandas(),
        outcome_timestamps=df["meta_time_of_diagnosis_fallback_nan"].to_pandas(),
        y=df["y"].to_pandas(),
        y_hat_probs=df["y_hat_prob"].to_pandas(),
        age=df["pred_age_in_years"].to_pandas(),
        is_female=df["pred_sex_female_layer_1"].to_pandas(),
        custom_columns={
            "scz_or_bp": df["meta_scz_or_bp_indicator_fallback_nan"].to_pandas(),
            "first_visit": df["meta_first_visit_fallback_nan"].to_pandas(),
            "time_of_scz_diagnosis": df["meta_time_of_scz_diagnosis_fallback_nan"].to_pandas(),
            "time_of_bp_diagnosis": df["meta_time_of_bp_diagnosis_fallback_nan"].to_pandas(),
        },
    )


def _load_validation_data_from_schema(schema: BaselineSchema) -> pl.DataFrame:
    match schema.trainer:
        case CrossValidatorTrainer() | SyntheticCrossValidatorTrainer():
            return schema.trainer.training_data.load().collect()
        case SplitTrainer() | SplitTrainerSeparatePreprocessing() | SyntheticSplitTrainerSeparatePreprocessing():
            return schema.trainer.validation_data.load().collect()
        case BaselineTrainer():
            raise TypeError("That's an ABC, mate")


def cohort_metadata_from_run(
    run: PsycopMlflowRun, cohort_metadata_cols: Sequence[pl.Expr]
) -> pl.DataFrame:
    cfg = run.get_config()
    schema = BaselineSchema(**BaselineRegistry.resolve(cfg))

    return _load_validation_data_from_schema(schema=schema).select(cohort_metadata_cols)


def load_sczbp_metadata() -> pl.DataFrame:
    return (
        pl.read_parquet(
            OVARTACI_SHARED_DIR
            / "scz_bp"
            / "flattened_datasets"
            / "metadata_only"
            / "metadata_only.parquet"
        )
        .drop("pred_time_uuid")
        .with_columns(pl.col("timestamp").dt.to_string(format="%Y-%m-%d-%H-%M-%S"))
        .with_columns(
            pl.concat_str("dw_ek_borger", "timestamp", separator="-").alias("pred_time_uuid")
        )
        .drop("dw_ek_borger", "timestamp")
    )


def scz_bp_get_eval_ds_from_best_run_in_experiment(experiment_name: str) -> EvalDataset:
    best_run = MlflowClientWrapper().get_best_run_from_experiment(
        experiment_name=experiment_name, metric="all_oof_BinaryAUROC"
    )

    # min_eval_ds = minimal_eval_dataset_from_mlflow_run(run=best_run) # noqa: ERA001
    min_eval_ds = minimal_eval_dataset_from_path(best_run.download_artifact("eval_df.parquet"))
    cohort_data = cohort_metadata_from_run(
        run=best_run,
        cohort_metadata_cols=[
            pl.col("prediction_time_uuid"),
            pl.col("dw_ek_borger"),
            pl.col("timestamp"),
            pl.col("pred_age_in_years"),
            pl.col("pred_sex_female_layer_1"),
        ],
    ).rename({"prediction_time_uuid": "pred_time_uuid"})
    cohort_metadata = load_sczbp_metadata()
    df = min_eval_ds.frame.join(
        cohort_data, how="left", on=min_eval_ds.pred_time_uuid_col_name
    ).join(cohort_metadata, how="left", on=min_eval_ds.pred_time_uuid_col_name, validate="1:1")
    return scz_bp_df_to_eval_df(df=df)


if __name__ == "__main__":
    experiment_name = "scz-bp/develop"
    best_run = MlflowClientWrapper().get_best_run_from_experiment(
        experiment_name=experiment_name, metric="all_oof_BinaryAUROC"
    )

    # min_eval_ds = minimal_eval_dataset_from_mlflow_run(run=best_run) # noqa: ERA001
    min_eval_ds = minimal_eval_dataset_from_path(
        Path(best_run.get_config()["project_info"]["experiment_path"]) / "eval_df.parquet"
    )
    cohort_metadata = cohort_metadata_from_run(
        run=best_run,
        cohort_metadata_cols=[
            pl.col("prediction_time_uuid"),
            pl.col("dw_ek_borger"),
            pl.col("timestamp"),
            pl.col("^meta.*$"),
            pl.col("pred_age_in_years"),
            pl.col("pred_sex_female_layer_1"),
        ],
    ).rename({"prediction_time_uuid": "pred_time_uuid"})
    df = min_eval_ds.frame.join(cohort_metadata, how="left", on=min_eval_ds.pred_time_uuid_col_name)
    eval_ds = scz_bp_df_to_eval_df(df=df)
