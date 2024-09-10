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
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    RegionalFilter, FilterByOutcomeStratifiedSplits
)
from psycop.common.types.validated_frame import ValidatedFrame
from psycop.projects.ect.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_ect_indicator,
)


@dataclass(frozen=True)
class TableOneModel(ValidatedFrame[pl.DataFrame]):
    outcome_col_name: str
    sex_col_name: str
    dataset_col_name: str = "dataset"
    pred_time_uuid_col_name: str = "pred_time_uuid"


@shared_cache().cache
def _train_test_column(flattened_data: pl.DataFrame) -> pl.DataFrame:
    """Adds a 'dataset' column to the dataframe, indicating whether the row is in the train or test set."""
    train_data = (
        FilterByOutcomeStratifiedSplits(["train", "val"])
        .apply(flattened_data.lazy())
        .with_columns(dataset=pl.lit("0. train"))
        .collect()
    )
    test_data = (
        FilterByOutcomeStratifiedSplits(["test"])
        .apply(flattened_data.lazy())
        .with_columns(dataset=pl.lit("test"))
        .collect()
    )

    flattened_combined = pl.concat([train_data, test_data], how="vertical").rename(
        {"prediction_time_uuid": "pred_time_uuid"}
    )
    return flattened_combined


def _preprocessed_data(data_path: str, pipeline: BaselinePreprocessingPipeline) -> pl.DataFrame:
    data = pl.scan_parquet(data_path)
    preprocessed = pipeline.apply(data)
    return pl.from_pandas(preprocessed)


@shared_cache().cache
def _first_outcome_data(data: pl.DataFrame) -> pl.DataFrame:
    first = get_first_ect_indicator()
    return data.join(pl.from_pandas(first), on="dw_ek_borger", how="left")


@shared_cache().cache
def table_one_model(run: PsycopMlflowRun, sex_col_name: str) -> TableOneModel:
    cfg = run.get_config()

    tmp_cfg = pathlib.Path(mkdtemp()) / "tmp.cfg"
    cfg.to_disk(tmp_cfg)

    from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry

    populate_baseline_registry()

    filled = resolve_and_fill_config(tmp_cfg, fill_cfg_with_defaults=True)

    pipeline: BaselinePreprocessingPipeline = filled["trainer"].preprocessing_pipeline
    pipeline._logger = TerminalLogger()  # type: ignore
    pipeline.steps = [
        step
        for step in pipeline.steps
        if "filter" in step.__class__.__name__.lower()  # Only keep potential row filters
        and "column"
        not in step.__class__.__name__.lower()  # Do not filter columns (e.g. keep timestamps for further processing)
        and "outcomestratified"
        not in step.__class__.__name__.lower()  # Do not filter on id split (happens when adding split labels)
    ]

    preprocessed_visits = _preprocessed_data(cfg["trainer"]["training_data"]["paths"][0], pipeline)
    split = _train_test_column(preprocessed_visits)
    with_outcome = _first_outcome_data(split)

    return TableOneModel(
        with_outcome.with_columns(
            (pl.col("pred_age_days_fallback_0") / 365.25).alias("pred_age_in_years")
        ),
        allow_extra_columns=True,
        outcome_col_name=cfg["trainer"]["outcome_col_name"],
        sex_col_name=sex_col_name,
    )
