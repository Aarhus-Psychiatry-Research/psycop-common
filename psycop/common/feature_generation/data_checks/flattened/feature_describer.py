"""Generates a df with feature descriptions for the predictors in the source
df."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from timeseriesflattener.v1.feature_specs.single_specs import PredictorSpec, StaticSpec
from wasabi import Printer

from psycop.common.feature_generation.data_checks.utils import save_df_to_pretty_html_table
from psycop.common.feature_generation.loaders.flattened.local_feature_loaders import load_split

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

UNICODE_HIST = {
    0: " ",
    1 / 8: "▁",
    1 / 4: "▂",
    3 / 8: "▃",
    1 / 2: "▄",
    5 / 8: "▅",
    3 / 4: "▆",
    7 / 8: "▇",
    1: "█",
}

HIST_BINS = 8


def get_value_proportion(series: pd.Series, value: Any) -> float:  # type: ignore
    """Get proportion of series that is equal to the value argument."""
    if np.isnan(value):
        return round(series.isna().mean(), 2)  # type: ignore
    return round(series.eq(value).mean(), 2)  # type: ignore


def _find_nearest(array: np.ndarray, value: Any) -> np.ndarray:  # type: ignore
    """Find the nearest numerical match to value in an array.

    Args:
        array (np.ndarray): An array of numbers to match with.
        value (float): Single value to find an entry in array that is close.

    Returns:
        np.array: The entry in array that is closest to value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def create_unicode_hist(series: pd.Series) -> pd.Series:  # type: ignore
    """Return a histogram rendered in block unicode. Given a pandas series of
    numerical values, returns a series with one entry, the original series
    name, and a histogram made up of unicode characters.

    Args:
        series (pd.Series): Numeric column of data frame for analysis

    Returns:
        pd.Series: Index of series name and entry with unicode histogram as
        a string, eg '▃▅█'

    All credit goes to the python package skimpy.
    """
    # Remove any NaNs
    series = series.dropna()

    if series.dtype == "bool":
        series = series.astype("int")  # type: ignore

    hist, _ = np.histogram(series, density=True, bins=HIST_BINS)
    hist = hist / hist.max()

    # Now do value counts
    key_vector = np.array(list(UNICODE_HIST.keys()), dtype="float")

    ucode_to_print = "".join([UNICODE_HIST[_find_nearest(key_vector, val)] for val in hist])

    return pd.Series(ucode_to_print)


def generate_temporal_feature_description(
    series: pd.Series,  # type: ignore
    predictor_spec: PredictorSpec,
    feature_name: str | None = None,
) -> dict[str, Any]:
    """Generate a row with feature description for a temporal predictor."""
    if feature_name is None:
        feature_name = predictor_spec.feature_base_name

    d = {
        "Predictor df": feature_name,
        "Lookbehind days": predictor_spec.lookbehind_days,
        "Resolve multiple": predictor_spec.aggregation_fn.__name__,
        "N unique": series.nunique(),
        "Fallback strategy": str(predictor_spec.fallback),
        "Proportion missing": series.isna().mean(),
        "Mean": round(series.mean(), 2),
        "Proportion using fallback": get_value_proportion(series, predictor_spec.fallback),
    }

    for percentile in (0.25, 0.5, 0.75):
        # Get the value representing the percentile
        d[f"{percentile * 100}-percentile"] = round(series.quantile(percentile), 1)

    return d


def generate_static_feature_description(
    series: pd.Series,  # type: ignore
    predictor_spec: StaticSpec,
) -> dict[str, Any]:
    """Generate a row with feature description for a static predictor."""
    return {
        "Predictor df": predictor_spec.feature_base_name,
        "Lookbehind days": "N/A",
        "Resolve multiple": "N/A",
        "N unique": series.nunique(),
        "Fallback strategy": "N/A",
        "Proportion missing": series.isna().mean(),
        "Mean": round(series.mean(), 2),
        "Proportion using fallback": "N/A",
    }


def generate_feature_description_row(
    series: pd.Series,  # type: ignore
    predictor_spec: StaticSpec | PredictorSpec,
    feature_name: str | None = None,
) -> dict[str, Any]:
    """Generate a row with feature description.

    Args:
        series (pd.Series): Series with data to describe.
        predictor_spec (PredictorSpec): Predictor specification.
        feature_name (str, optional): Name of the feature. Defaults to None.

    Returns:
        dict: dictionary with feature description.
    """

    match predictor_spec:
        case StaticSpec():
            return generate_static_feature_description(series, predictor_spec)
        case PredictorSpec():
            return generate_temporal_feature_description(
                series, predictor_spec, feature_name=feature_name
            )
        case _:  # type: ignore
            raise ValueError(f"Unknown predictor spec type: {type(predictor_spec)}")


def generate_feature_description_df(
    df: pd.DataFrame,
    predictor_specs: list[PredictorSpec | StaticSpec],
    prefixes_to_describe: set[str],
) -> pd.DataFrame:
    """Generate a data frame with feature descriptions.

    Args:
        df (pd.DataFrame): Data frame with data to describe.
        predictor_specs (Union[PredictorSpec, StaticSpec,]): Predictor specifications.
        prefixes_to_describe: Which prefixes for column names to make feature descriptions for.

    Returns:
        pd.DataFrame: Data frame with feature descriptions.
    """

    rows = []

    for spec in predictor_specs:
        column_name = spec.get_output_col_name()

        if spec.prefix in prefixes_to_describe:
            rows.append(
                generate_feature_description_row(series=df[column_name], predictor_spec=spec)
            )

    # Convert to dataframe
    feature_description_df = pd.DataFrame(rows)

    # Sort feature_description_df by Predictor df to make outputs easier to read
    feature_description_df = feature_description_df.sort_values(by="Predictor df")

    return feature_description_df


def save_feature_descriptive_stats_from_dir(
    feature_set_dir: Path,
    feature_specs: list[PredictorSpec | StaticSpec],
    file_suffix: str,
    prefixes_to_describe: set[str],
    splits: Sequence[str] = ("train",),
    out_dir: Path | None = None,
):
    """Write a html table and csv with descriptive stats for features in the directory.

    Args:
        feature_set_dir (Path): Path to directory with data frames.
        feature_specs (list[PredictorSpec]): List of feature specifications.
        file_suffix (str): Suffix of the data frames to load. Must be either ".csv" or ".parquet".
        prefixes_to_describe: Which prefixes for column names to make feature descriptions for.
        splits (tuple[str]): tuple of splits to include in the description. Defaults to ("train").
        out_dir (Path): Path to directory where to save the feature description. Defaults to None.
    """
    msg = Printer(timestamp=True)

    if out_dir is None:
        out_dir = feature_set_dir / "feature_set_descriptive_stats"

    out_dir.mkdir(exist_ok=True, parents=True)

    for split in splits:
        msg.info(f"{split}: Creating descriptive stats for feature set")

        dataset = load_split(feature_set_dir=feature_set_dir, split=split, file_suffix=file_suffix)

        msg.info(f"{split}: Generating descriptive stats dataframe")

        feature_descriptive_stats = generate_feature_description_df(
            df=dataset, predictor_specs=feature_specs, prefixes_to_describe=prefixes_to_describe
        )

        msg.info(f"{split}: Writing descriptive stats dataframe to disk")

        feature_descriptive_stats.to_csv(
            out_dir / f"{split}_feature_descriptive_stats.csv", index=False
        )
        # Writing html table as well
        save_df_to_pretty_html_table(
            df=feature_descriptive_stats,
            path=out_dir / f"{split}_feature_descriptive_stats.html",
            title="Feature descriptive stats",
        )
