import re
from pathlib import Path
from typing import Callable

import polars as pl
import polars.selectors as cs
from confection import Config

from psycop.common.feature_generation.data_checks.flattened.feature_describer_tsflattener_v2 import (
    ParsedPredictorColumn,
    generate_feature_description_df,
    parse_predictor_column_name,
    tsflattener_v2_column_is_static,
)
from psycop.common.feature_generation.data_checks.utils import save_df_to_pretty_html_table
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.loggers.dummy_logger import DummyLogger


def get_filtered_prediction_times(cfg: Config) -> pl.DataFrame:
    data = BaselineRegistry.resolve({"data": cfg["trainer"]["training_data"]})["data"].load()

    preprocessing_pipeline = BaselineRegistry().resolve(
        {"pipe": cfg["trainer"]["preprocessing_pipeline"]}
    )["pipe"]
    preprocessing_pipeline._logger = DummyLogger()
    preprocessed_all_splits: pl.DataFrame = pl.from_pandas(preprocessing_pipeline.apply(data))

    return preprocessed_all_splits


def parse_predictor_column_name_v1(
    col_name: str,
    is_static: Callable[[str], bool] = tsflattener_v2_column_is_static,
    feature_name_regex: str = r"[a-zA-Z]+_(.*?)_(?=fallback|within)",
    fallback_regex: str = r"_fallback_(.+)$",
    time_interval_end_regex: str = "_within_([0-9]+)_",
    time_interval_format_regex: str = r"within_[0-9]+_([a-z]+)_",
    resolve_multiple_strategy_regex: str = r"([a-z]+)_fallback",
) -> ParsedPredictorColumn:
    col_is_static = is_static(col_name)
    feature_name = re.search(feature_name_regex, col_name).group(1)  # type: ignore
    fallback = re.search(fallback_regex, col_name).group(1)  # type: ignore
    time_interval_start = "0"
    time_interval_end = (
        re.search(time_interval_end_regex, col_name).group(1) if not col_is_static else "N/A"  # type: ignore
    )
    time_interval_format = (
        re.search(time_interval_format_regex, col_name).group(1) if not col_is_static else "N/A"  # type: ignore
    )
    resolve_multiple_strategy = (
        re.search(resolve_multiple_strategy_regex, col_name).group(1)  # type: ignore
        if not col_is_static
        else "N/A"
    )
    return ParsedPredictorColumn(
        col_name=col_name,
        feature_name=feature_name,
        fallback=fallback,
        time_interval_start=time_interval_start,
        time_interval_end=time_interval_end,
        time_interval_format=time_interval_format,
        resolve_multiple_strategy=resolve_multiple_strategy,
        is_static=is_static(col_name),
    )


if __name__ == "__main__":
    cfg = Config().from_disk(Path(__file__).parent / "predictor_description.cfg")

    populate_baseline_registry()

    filtered_dataset = (
        get_filtered_prediction_times(cfg)
        .select(cs.starts_with("pred_"))
        .rename({"pred_adm_day_count": "pred_admission_day_fallback_nan"})
    )

    predictor_description_df = generate_feature_description_df(
        df=filtered_dataset,
        column_name_parser=lambda c: parse_predictor_column_name(
            c,
            time_interval_end_regex=r"_within(.*?)[0-9]+",
            time_interval_format_regex=r"_within_[0-9]+_([a-z]+)",
        ),
    )
    predictor_description_df.write_csv(
        Path(__file__).parent / "tables" / "predictor_description.html"
    )
