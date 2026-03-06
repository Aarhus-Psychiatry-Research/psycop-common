import copy
import pathlib
from dataclasses import dataclass
from tempfile import mkdtemp
from typing import Literal

import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.common.model_training_v2.config.config_utils import PsycopConfig, resolve_and_fill_config
from psycop.common.model_training_v2.loggers.terminal_logger import TerminalLogger
from psycop.common.model_training_v2.trainer.preprocessing.pipeline import (
    BaselinePreprocessingPipeline,
)

from psycop.common.types.validated_frame import ValidatedFrame
from psycop.projects.t2d_extended.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_diabetes_lab_result_above_threshold,
)


@dataclass(frozen=True)
class TableOneModel(ValidatedFrame[pl.DataFrame]):
    outcome_col_name: str
    sex_col_name: str
    dataset_col_name: str = "dataset"
    pred_time_uuid_col_name: str = "pred_time_uuid"


# @shared_cache().cache
# def _train_test_column(flattened_data: pl.DataFrame) -> pl.DataFrame:
#     """Adds a 'dataset' column to the dataframe, indicating whether the row is in the train or test set."""
#     train_data = (
#         RegionalFilter(["train", "val"])
#         .apply(flattened_data.lazy())
#         .with_columns(dataset=pl.lit("0. train"))
#         .collect()
#     )
#     test_data = (
#         RegionalFilter(["test"])
#         .apply(flattened_data.lazy())
#         .with_columns(dataset=pl.lit("test"))
#         .collect()
#     )

#     flattened_combined = pl.concat([train_data, test_data], how="vertical").rename(
#         {"prediction_time_uuid": "pred_time_uuid"}
#     )
#     return flattened_combined


def _add_dataset_column(flattened_data: pl.DataFrame, dataset_name: str) -> pl.DataFrame:
    """Adds a 'dataset' column to the dataframe, indicating whether the row is in the train or test set."""
    flattened_data_with_dataset_column = flattened_data.with_columns(pl.lit(f"{dataset_name}").alias("dataset"))

    return flattened_data_with_dataset_column



def _preprocessed_data(data_path: str, pipeline: BaselinePreprocessingPipeline) -> pl.DataFrame:
    data = pl.scan_parquet(data_path)
    preprocessed = pipeline.apply(data)
    return pl.from_pandas(preprocessed)


@shared_cache().cache
def _first_outcome_data(data: pl.DataFrame) -> pl.DataFrame:
    first = get_first_diabetes_lab_result_above_threshold()
    return data.join(pl.from_pandas(first), on="dw_ek_borger", how="left")



@shared_cache().cache
def prepare_table_one_dataset(cfg: PsycopConfig, sex_col_name: str, dataset_name: str, split: Literal["train", "val"], ) -> pl.DataFrame:

    tmp_cfg = pathlib.Path(mkdtemp()) / "tmp.cfg"
    cfg.to_disk(tmp_cfg)

    from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
    from psycop.projects.t2d_extended.model_training.populate_t2d_registry import (
        populate_with_t2d_extended_registry,
    )

    populate_baseline_registry()
    populate_with_t2d_extended_registry()

    filled = resolve_and_fill_config(tmp_cfg, fill_cfg_with_defaults=True)

    if split == "train":
        pipeline: BaselinePreprocessingPipeline = filled["trainer"].training_preprocessing_pipeline
        data_path = cfg["trainer"]["training_data"]["paths"][0]
    else:
        pipeline = filled["trainer"].validation_preprocessing_pipeline
        data_path = cfg["trainer"]["validation_data"]["paths"][0]

    pipeline._logger = TerminalLogger()  # type: ignore
    pipeline = copy.deepcopy(pipeline)
    pipeline.steps = [
        step
        for step in pipeline.steps
        if "filter" in step.__class__.__name__.lower()  # Only keep potential row filters
        and "column"
        not in step.__class__.__name__.lower()  # Do not filter columns (e.g. keep timestamps for further processing)
        and "regional"
        not in step.__class__.__name__.lower()  # Do not filter on region (happens when adding split labels)
    ]

    preprocessed_visits = _preprocessed_data(data_path, pipeline)

    df = _add_dataset_column(preprocessed_visits, f"{dataset_name}").rename(
        {"prediction_time_uuid": "pred_time_uuid"}
    )

    df = _first_outcome_data(df)

    df = df.with_columns(
            (pl.col("pred_age_days_fallback_0") / 365.25).alias("pred_age_in_years")
        )
    
    return df
