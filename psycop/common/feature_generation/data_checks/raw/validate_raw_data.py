"""Functions for validating raw data - in the sense of data returned from a
loader."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from wasabi import Printer

from psycop.common.feature_generation.data_checks.flattened.feature_describer import (
    create_unicode_hist,
)
from psycop.common.feature_generation.data_checks.utils import save_df_to_pretty_html_table

if TYPE_CHECKING:
    from pathlib import Path


def median_absolute_deviation(series: pd.Series) -> float:  # type: ignore
    """Calculates a series' median absolute deviation from its own median.

    Args:
        series (pd.Series): Series to calculate the median absolute deviation of.

    Returns:
        np.array: Median absolute deviation of the series.
    """
    med = np.median(series)
    return np.median(np.abs(series - med))  # type: ignore


def generate_column_description(series: pd.Series) -> dict:  # type: ignore
    """Generates a dictionary with column description.

    Args:
        series (pd.Series): Series to describe.

    Returns:
        dict: dictionary with column description.
    """

    d = {
        "col_name": series.name,
        "dtype": series.dtype,
        "n_unique": series.nunique(),
        "n_missing": series.isna().sum(),
        "min": series.min(),
        "max": series.max(),
        "mean": series.mean(),
        "std": series.std(),
        "median": series.median(),
        "median_absolute_deviation": median_absolute_deviation(series),
    }
    d["histogram"] = create_unicode_hist(series)
    for percentile in [0.01, 0.25, 0.5, 0.75, 0.99]:
        d[f"{percentile}th_percentile"] = round(series.quantile(percentile), 1)

    return d


def get_na_prob(series: pd.Series) -> float:  # type: ignore
    """Calculates the propotion of rows that are NaT.

    Args:
        series (pd.Series): Series of timestamps

    Returns:
        pd.Series: Series with proportion of NaT
    """
    return series.isna().sum() / len(series)


def highlight_large_deviation(
    series: pd.Series,  # type: ignore
    threshold_ratio: float,
    baseline_column: str,
    variation_column: str,
) -> list[str]:
    """Highlights rows where the 99th/1st percentile is x times the variation
    column larger/smaller than the baseline column (probably mean or median).

    Args:
        series (pd.Series): Series to describe.
        threshold_ratio (float): Threshold for deviation. 3-4 might be a good value.
        baseline_column (str): Name of the column to use as baseline. Commonly 'mean' or 'median'.
        variation_column (str): Name of the column containing the variation.
        Commonly 'std' or 'median_absolute_deviation'

    Returns:
        list[str]: list of styles for each row.
    """
    above_threshold = pd.Series(data=False, index=series.index)
    lower_bound = series[baseline_column] - series[variation_column] * threshold_ratio
    upper_bound = series[baseline_column] + series[variation_column] * threshold_ratio

    above_threshold[baseline_column] = (
        series.loc["0.99th_percentile"] > upper_bound
        or series.loc["0.01th_percentile"] < lower_bound
    )
    return ["background-color: yellow" if above_threshold.any() else "" for _ in above_threshold]


def validate_raw_data(
    save_path: Path,
    df: pd.DataFrame,
    feature_set_name: str,
    deviation_baseline_column: str | None = "median",
    deviation_threshold_ratio: float | None = 4.0,
    deviation_variation_column: str | None = "median_absolute_deviation",
) -> None:
    """Validates raw data from SQL database (or any dataframe, really). Calculates summary statistics.
    Summary statistics are saved as a table with one row for each column. Rows
    are colored yellow if the 99th/1st percentile exceeds.

    `deviation_baseline_column'  +- `deviation_treshold` * `deviation_variation_column`.
    All files are saved to the `RAW_DATA_VALIDATION_PATH` directory in a subdirectory
    named `feature_set_name`.

    Args:
        save_path (Path): Path to save the validation results to.
        df (pd.DataFrame): Dataframe to validate.
        feature_set_name (str): Name of the feature set.
        deviation_baseline_column (Optional[str], optional): Which metric to use as
            the baseline for identifying columns that might be problematic. Defaults to "median".
        deviation_threshold_ratio (Optional[float], optional): Which threshold to
            use as the cutoff for identifying problematic columns. Defaults to 4.0.
        deviation_variation_column (Optional[str], optional): Which metric to use
            as the measure of variation from the baseline when identifying problematic
            columns. Defaults to "median_absolute_deviation".

    Raises:
        ValueError: If 'dw_ek_borger' or 'timestamp' are not columns in the dataframe.
    """

    msg = Printer(timestamp=True)
    failed_checks = {}

    save_path.mkdir(parents=True, exist_ok=True)

    # check if `timestamp` and `dw_ek_borger` columns exist
    timestamp_col_name = "timestamp" if "timestamp" in df.columns else None
    id_col_name = "dw_ek_borger" if "dw_ek_borger" in df.columns else None
    if timestamp_col_name or id_col_name is None:
        raise ValueError("Dataframe must contain `timestamp` and `dw_ek_borger` columns.")

    # Data description
    data_columns = [col for col in df.columns if col not in [id_col_name, timestamp_col_name]]
    with msg.loading("Generating data description..."):
        data_description = [generate_column_description(df[col]) for col in data_columns]

    msg.good("Finished data description.")

    data_description = pd.DataFrame(data_description)
    data_description["prop NaT"] = get_na_prob(df[timestamp_col_name])  # type: ignore
    data_description.to_csv(save_path / "data_description.csv", index=False)  # type: ignore
    # Highlight rows with large deviations from the baseline
    data_description = data_description.style.apply(  # type: ignore
        func=highlight_large_deviation,  # type: ignore
        threshold=deviation_threshold_ratio,
        baseline_column=deviation_baseline_column,
        variation_column=deviation_variation_column,
        axis=1,  # type: ignore
    )

    save_df_to_pretty_html_table(
        data_description,  # type: ignore
        save_path / "data_description.html",
        title=f"Data description - {feature_set_name}",
        subtitle=f"Yellow rows indicate {deviation_threshold_ratio}x deviations from the {deviation_baseline_column}\n(99th/1st percentile within {deviation_baseline_column} +- {deviation_variation_column} * threshold={deviation_threshold_ratio}) from the baseline.)",
    )

    msg.info(f"All files saved to {save_path}")
    if failed_checks:
        print(f"The following checks failed - look through the generated reports!\n{failed_checks}")
