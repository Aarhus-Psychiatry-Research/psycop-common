import re
from dataclasses import dataclass
from typing import Callable

import polars as pl
from rich.pretty import pprint

from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR


@dataclass
class ParsedPredictorColumn:
    col_name: str
    feature_name: str
    fallback: str
    time_interval_start: str
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
    fallback_regex: str = r"_fallback_(\d+|nan)",
    time_interval_start_regex: str = r"_within_([0-9]+)",
    resolve_multiple_strategy_regex: str = r"([a-z]+)_fallback",
) -> ParsedPredictorColumn:
    if is_static(col_name):
        return ParsedPredictorColumn(
            col_name=col_name,
            feature_name=re.search(r"[a-zA-Z]+_(.+)", col_name).group(1),  # type: ignore
            fallback="0",
            time_interval_start="N/A",
            resolve_multiple_strategy="N/A",
            is_static=True,
        )

    feature_name = _get_match_group(feature_name_regex, col_name)
    fallback = _get_match_group(fallback_regex, col_name)
    time_interval_start = _get_match_group(time_interval_start_regex, col_name)
    resolve_multiple_strategy = _get_match_group(resolve_multiple_strategy_regex, col_name)

    return ParsedPredictorColumn(
        col_name=col_name,
        feature_name=feature_name,
        fallback=fallback,
        time_interval_start=time_interval_start,
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
        mean = round(df[parsed_col.col_name].drop_nans().mean(), 3)  # type: ignore
        if parsed_col.fallback == "nan":
            proportion_using_fallback = round(df[parsed_col.col_name].null_count() / len(df), 3)
        else:
            proportion_using_fallback = round(
                df[parsed_col.col_name].cast(pl.Float32).eq(float(parsed_col.fallback)).mean(),  # type: ignore
                3,
            )

        if "tfidf" in parsed_col.feature_name:
            vocab = pl.read_parquet(
                OVARTACI_SHARED_DIR
                / "text_models"
                / "vocabulary_lists"
                / "vocab_tfidf_psycop_train_all_sfis_preprocessed_sfi_type_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750.parquet"
            )

            tfidf_idx = re.search(r"(\d+)", parsed_col.feature_name).group(0)  # type: ignore
            tfidf_word = vocab.filter(pl.col("Index") == int(tfidf_idx))["Word"][0]
            parsed_col.feature_name = re.sub(tfidf_idx, tfidf_word, parsed_col.feature_name)

        feature_description = {
            "Feature name": parsed_col.feature_name,
            "Lookbehind period": parsed_col.time_interval_start
            if parsed_col.is_static
            else parsed_col.time_interval_start,
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
