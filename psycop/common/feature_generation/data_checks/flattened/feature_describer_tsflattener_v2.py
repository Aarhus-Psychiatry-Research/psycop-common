import re
from dataclasses import dataclass
from typing import Any, Callable

import polars as pl
from rich.pretty import pprint
from torch import Value


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


def _get_match_group(regex: str, string: str) -> str:
    match = re.search(regex, string)
    if match is None:
        raise ValueError(
            f"Column '{string}' did not match regex: '{regex}'. All temporal columns should match, and column is not identified as static. Either adjust regex to match, or adjust static/temporal rule."
        )
    return match.group(1)


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
    feature_name = _get_match_group(feature_name_regex, col_name)
    fallback = _get_match_group(fallback_regex, col_name)

    if is_static(col_name):
        return ParsedPredictorColumn(
            col_name=col_name,
            feature_name=feature_name,
            fallback=fallback,
            time_interval_start="N/A",
            time_interval_end="N/A",
            time_interval_format="N/A",
            resolve_multiple_strategy="N/A",
            is_static=True,
        )

    time_interval_start = _get_match_group(time_interval_start_regex, col_name)
    time_interval_end = _get_match_group(time_interval_end_regex, col_name)
    time_interval_format = _get_match_group(time_interval_format_regex, col_name)
    resolve_multiple_strategy = _get_match_group(resolve_multiple_strategy_regex, col_name)

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
    errors = []
    parsed_cols: list[ParsedPredictorColumn] = []
    for col in df.columns:
        try:
            parsed_cols.append(column_name_parser(col))
        except ValueError as e:
            errors.append(e)
            continue

    if errors:
        pprint(errors)
        raise ValueError(errors)

    feature_rows = []
    for parsed_col in parsed_cols:
        n_unique = df[parsed_col.col_name].n_unique()
        mean = round(df[parsed_col.col_name].drop_nans().mean(), 2)  # type: ignore
        proportion_using_fallback = round(
            df[parsed_col.col_name].cast(pl.Float32).eq(float(parsed_col.fallback)).mean(), 2
        )  # type: ignore

        feature_description = {
            "Feature name": parsed_col.feature_name,
            "Lookbehind period": parsed_col.time_interval_start
            if parsed_col.is_static
            else f"{parsed_col.time_interval_start} to {parsed_col.time_interval_end} {parsed_col.time_interval_format}",
            "Resolve multiple": parsed_col.resolve_multiple_strategy,
            "Fallback strategy": parsed_col.fallback,
            "Static": parsed_col.is_static,
            "Mean": mean,
            "N. unique": n_unique,
            "Proportion using fallback": proportion_using_fallback,
        }
        feature_rows.append(feature_description)

    feature_description_df = pl.DataFrame(feature_rows)
    return feature_description_df.sort("Feature name")
