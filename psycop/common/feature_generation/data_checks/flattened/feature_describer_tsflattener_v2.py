import re
from dataclasses import dataclass
from typing import Any, Callable

import polars as pl


@dataclass
class ParsedPredictorColumn:
    col_name: str
    feature_name: str
    fallback: str
    time_interval_start: str
    time_interval_end: str
    time_interval_format: str
    resolve_multiple_strategy: str
    is_static: bool


def tsflattener_v2_column_is_static(col_name: str) -> bool:
    str_match = "within"
    return str_match not in col_name


def parse_predictor_column_name(
    col_name: str,
    is_static: Callable[[str], bool] = tsflattener_v2_column_is_static,
    feature_name_regex: str = r"[a-zA-Z]+_(.*?)_(?=fallback|within)",
    fallback_regex: str = r"_fallback_(.+)$",
    time_interval_start_regex: str = r"_within_([0-9]+)",
    time_interval_end_regex: str = "_to_([0-9]+)_",
    time_interval_format_regex: str = r"to_[0-9]+_([a-z]+)_",
    resolve_multiple_strategy_regex: str = r"([a-z]+)_fallback",
) -> ParsedPredictorColumn:
    col_is_static = is_static(col_name)
    feature_name = re.search(feature_name_regex, col_name).group(1)  # type: ignore
    fallback = re.search(fallback_regex, col_name).group(1)  # type: ignore
    time_interval_start = (
        re.search(time_interval_start_regex, col_name).group(1) if not col_is_static else "N/A"  # type: ignore
    )
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


def generate_feature_description_df(
    df: pl.DataFrame,
    column_name_parser: Callable[[str], ParsedPredictorColumn] = parse_predictor_column_name,
) -> pl.DataFrame:
    feature_rows = []
    for col in df.columns:
        parsed_col = column_name_parser(col)
        n_unique = df[col].n_unique()
        proportion_missing = round(df[col].is_nan().mean(), 2)  # type: ignore
        mean = round(df[col].drop_nans().mean(), 2)  # type: ignore
        proportion_using_fallback = round(df[col].eq(float(parsed_col.fallback)).mean(), 2)  # type: ignore

        feature_description = {
            "Feature name": parsed_col.feature_name,
            "Lookbehind period": parsed_col.time_interval_start
            if parsed_col.is_static
            else f"{parsed_col.time_interval_start} to {parsed_col.time_interval_end} {parsed_col.time_interval_format}",
            "Resolve multiple": parsed_col.resolve_multiple_strategy,
            "Fallback strategy": parsed_col.fallback,
            "Static": parsed_col.is_static,
            "Proportion missing": proportion_missing,
            "Mean": mean,
            "N. unique": n_unique,
            "Proportion using fallback": proportion_using_fallback,
        }
        feature_rows.append(feature_description)
    feature_description_df = pl.DataFrame(feature_rows)
    return feature_description_df.sort("Feature name")
