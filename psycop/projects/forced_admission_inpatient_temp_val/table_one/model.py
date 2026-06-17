import pathlib
from dataclasses import dataclass
from tempfile import mkdtemp

import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.common.global_utils.mlflow.mlflow_data_extraction import PsycopMlflowRun
from psycop.common.model_training_v2.config.config_utils import resolve_and_fill_config
from psycop.common.model_training_v2.loggers.terminal_logger import TerminalLogger
from psycop.common.model_training_v2.trainer.preprocessing.pipeline import (
    BaselinePreprocessingPipeline,
)
from psycop.common.types.validated_frame import ValidatedFrame


@dataclass(frozen=True)
class TableOneModel(ValidatedFrame[pl.DataFrame]):
    outcome_col_name: str
    sex_col_name: str
    dataset_col_name: str = "dataset"
    pred_time_uuid_col_name: str = "pred_time_uuid"


# -----------------------------
# Helpers
# -----------------------------


def _preprocessed_data(data_path: str, pipeline: BaselinePreprocessingPipeline) -> pl.DataFrame:
    """Load and apply preprocessing pipeline."""
    data = pl.scan_parquet(data_path)
    processed = pipeline.apply(data)

    processed = pl.from_pandas(processed)

    return processed


def _load_with_dataset_label(
    data_path: str, pipeline: BaselinePreprocessingPipeline, dataset_name: str
) -> pl.DataFrame:
    """Load, preprocess and attach dataset label."""
    df = _preprocessed_data(data_path, pipeline)
    return df.with_columns(pl.lit(dataset_name).alias("dataset"))


# -----------------------------
# Main entry
# -----------------------------


@shared_cache().cache
def table_one_model(run: PsycopMlflowRun, sex_col_name: str) -> TableOneModel:
    cfg = run.get_config()

    # Write config to temp file for resolver
    tmp_cfg = pathlib.Path(mkdtemp()) / "tmp.cfg"
    cfg.to_disk(tmp_cfg)

    from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry

    populate_baseline_registry()

    filled = resolve_and_fill_config(tmp_cfg, fill_cfg_with_defaults=True)

    # -----------------------------
    # Pipelines from config
    # -----------------------------
    training_pipeline: BaselinePreprocessingPipeline = filled[
        "trainer"
    ].training_preprocessing_pipeline
    validation_pipeline: BaselinePreprocessingPipeline = filled[
        "trainer"
    ].validation_preprocessing_pipeline

    # Optional logging
    training_pipeline._logger = TerminalLogger()  # type: ignore
    validation_pipeline._logger = TerminalLogger()  # type: ignore

    # -----------------------------
    # Remove column-filtering steps that would break Table 1
    # -----------------------------
    def _clean_pipeline(pipeline: BaselinePreprocessingPipeline) -> BaselinePreprocessingPipeline:
        pipeline.steps = [
            step for step in pipeline.steps if "column" not in step.__class__.__name__.lower()
        ]
        return pipeline

    training_pipeline = _clean_pipeline(training_pipeline)
    validation_pipeline = _clean_pipeline(validation_pipeline)

    # -----------------------------
    # Paths from config
    # -----------------------------
    train_path = cfg["trainer"]["training_data"]["paths"][0]
    val_path = cfg["trainer"]["validation_data"]["paths"][0]

    # -----------------------------
    # Load data (REAL config split)
    # -----------------------------
    train_df = _load_with_dataset_label(train_path, training_pipeline, "train_val")
    val_df = _load_with_dataset_label(val_path, validation_pipeline, "temp_val")

    combined = pl.concat([train_df, val_df], how="vertical")

    # -----------------------------
    # Feature engineering for Table 1
    # -----------------------------
    if "pred_age_days_fallback_0" in combined.columns:
        combined = combined.with_columns(
            (pl.col("pred_age_days_fallback_0") / 365.25).alias("pred_age_in_years")
        )

    combined = combined.rename({"prediction_time_uuid": "pred_time_uuid"})

    # -----------------------------
    # Build Table One model
    # -----------------------------
    return TableOneModel(
        combined,
        allow_extra_columns=True,
        outcome_col_name=cfg["trainer"]["training_outcome_col_name"],
        sex_col_name=sex_col_name,
    )
