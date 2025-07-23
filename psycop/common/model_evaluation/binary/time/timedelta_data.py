from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from psycop.common.model_evaluation.binary.utils import auroc_by_group, sensitivity_by_group
from psycop.common.model_evaluation.utils import bin_continuous_data, round_floats_to_edge

TIMEDELTA_STRINGS = Literal["h", "D", "M", "Q", "Y"]


def get_timedelta_series(
    direction: Literal["t1-t2", "t2-t1"],
    bin_unit: TIMEDELTA_STRINGS,
    df: pd.DataFrame,
    t2_col_name: str,
    t1_col_name: str,
) -> pd.Series:  # type: ignore
    """Calculate the time difference between two timestamps."""
    if direction == "t1-t2":
        df["unit_from_event"] = (df[t1_col_name] - df[t2_col_name]) / np.timedelta64(1, bin_unit)  # type: ignore
    elif direction == "t2-t1":
        df["unit_from_event"] = (df[t2_col_name] - df[t1_col_name]) / np.timedelta64(1, bin_unit)  # type: ignore
    else:
        raise ValueError(f"Direction should be one of ['t1-t2', 't2-t1'], not {direction}")

    return df["unit_from_event"]


def get_timedelta_df(
    y: Iterable[int],
    output: Iterable[float],
    time_one: Iterable[pd.Timestamp],
    time_two: Iterable[pd.Timestamp],
    direction: Literal["t1-t2", "t2-t1"],
    bins: Sequence[float],
    bin_unit: TIMEDELTA_STRINGS,
    bin_continuous_input: bool = True,
    drop_na_events: bool = True,
    min_n_in_bin: int = 5,
) -> pd.DataFrame:
    """Get a timedelta dataframe with the time difference between two timestamps.

    Args:
        y (Iterable[int]): True labels
        output (Iterable[float]): Model predictions. This can be either be probabilities or predictions, depending on whether the timedelta dataframe will be used for calculating sensitivity or AUROC.
        time_one (Iterable[pd.Timestamp]): Timestamp for time one
        time_two (Iterable[pd.Timestamp]): Timestamp for time two
        direction (Literal[&quot;t1): The order of the timestamps in the calculation of time difference. Takes either "t1-t2" or "t2-t1"
        bins (Sequence[float]): Bin values to aggregate by.
        bin_unit (TIMEDELTA_STRINGS): The unit of bin values, such as hours or days. Takes "h", "D", "M", "Q", or "Y".
        bin_continuous_input (bool, optional): Whether to aggregate continuous output by bins. Defaults to True.
        drop_na_events (bool, optional): Whether to drop missing values from the first timestamp. Defaults to True.
        min_n_in_bin (int, optional): Minimum number of samples in included bins. If there are less than n samples in a bin, it will be set to NA. Defaults to 5.

    Returns:
        pd.DataFrame: Timedelta dataframe
    """

    df = pd.DataFrame(
        {"y": y, "output": output, "t1_timestamp": time_one, "t2_timestamp": time_two}
    )
    # Drop rows with no events if specified
    if drop_na_events:
        df = df.dropna(subset=["t1_timestamp"])

    # Calculate difference in days between prediction and event
    df["unit_from_event"] = get_timedelta_series(
        direction=direction,
        bin_unit=bin_unit,
        df=df,
        t2_col_name="t2_timestamp",
        t1_col_name="t1_timestamp",
    )

    # bin data
    if bin_continuous_input:
        # Convert df["unit_from_event"] to int if possible
        df["unit_from_event_binned"], df["n_in_bin"] = bin_continuous_data(
            df["unit_from_event"], bins=bins, min_n_in_bin=min_n_in_bin
        )
    else:
        df["unit_from_event_binned"] = round_floats_to_edge(df["unit_from_event"], bins=bins)

    return df


def get_auroc_by_timedelta_df(
    y: Iterable[int],
    y_hat_probs: Iterable[float],
    time_one: Iterable[pd.Timestamp],
    time_two: Iterable[pd.Timestamp],
    direction: Literal["t1-t2", "t2-t1"],
    bins: Sequence[float],
    bin_unit: TIMEDELTA_STRINGS,
    confidence_interval: bool = True,
    bin_continuous_input: bool = True,
    drop_na_events: bool = True,
    stratified: bool = False,
    min_n_in_bin: int = 5,
    n_bootstraps: int = 100,
) -> pd.DataFrame:
    """Create dataframe for plotting performance metric from time to or from
    some event (e.g. time of diagnosis, time from first visit).

    Args:
        y: True labels
        y_hat_probs: The predicted probabilities by the function.
        metric_fn: Function to calculate metric
        time_one: Timestamps for time one (e.g. first visit).
        time_two: Timestamps for time two.
        direction: Which direction to calculate time difference.
        Can either be 't2-t1' or 't1-t2'.
        bins: Bins to group by.
        confidence_interval: Confidence interval to use for
        bin_unit: Unit of time to use for bins.
        bin_continuous_input: Whether to bin input. Defaults to True.
        drop_na_events: Whether to drop rows where the event is NA. Defaults to True.
        stratified: Whether to use stratified bootstrapping for confidence intervals.
        min_n_in_bin: Minimum number of rows in a bin to include in output. Defaults to 10.
        n_bootstraps: number of samples for bootstrap resampling
    Returns:
        pd.DataFrame: Dataframe ready for plotting where each row represents a bin.
    """
    df = get_timedelta_df(
        y=y,
        output=y_hat_probs,
        time_one=time_one,
        time_two=time_two,
        direction=direction,
        bins=bins,
        bin_unit=bin_unit,
        bin_continuous_input=bin_continuous_input,
        drop_na_events=drop_na_events,
        min_n_in_bin=min_n_in_bin,
    ).rename(columns={"output": "y_hat_probs"})

    df = df.rename({"y_hat": "y_hat_probs"}, axis=1)

    grouped_df = auroc_by_group(
        df=df,
        groupby_col_name="unit_from_event_binned",
        confidence_interval=confidence_interval,
        n_bootstraps=n_bootstraps,
        stratified=stratified,
    )

    return grouped_df


def get_sensitivity_by_timedelta_df(
    y: Iterable[int],
    y_hat: Iterable[float],
    time_one: Iterable[pd.Timestamp],
    time_two: Iterable[pd.Timestamp],
    direction: Literal["t1-t2", "t2-t1"],
    bins: Sequence[float],
    bin_unit: TIMEDELTA_STRINGS,
    confidence_interval: bool = True,
    bin_continuous_input: bool = True,
    drop_na_events: bool = True,
    stratified: bool = False,
    min_n_in_bin: int = 5,
) -> pd.DataFrame:
    """Create dataframe for plotting performance metric from time to or from
    some event (e.g. time of diagnosis, time from first visit).

    Args:
        y: True labels
        y_hat: The predicted class.
        metric_fn: Function to calculate metric
        time_one: Timestamps for time one (e.g. first visit).
        time_two: Timestamps for time two.
        direction: Which direction to calculate time difference.
        Can either be 't2-t1' or 't1-t2'.
        bins: Bins to group by.
        confidence_interval: Confidence interval to use for
        bin_unit: Unit of time to use for bins.
        bin_continuous_input: Whether to bin input. Defaults to True.
        drop_na_events: Whether to drop rows where the event is NA. Defaults to True.
        stratified: Whether to use stratified bootstrapping for confidence intervals.
        min_n_in_bin: Minimum number of rows in a bin to include in output. Defaults to 10.
    Returns:
        pd.DataFrame: Dataframe ready for plotting where each row represents a bin.
    """
    df = get_timedelta_df(
        y=y,
        output=y_hat,
        time_one=time_one,
        time_two=time_two,
        direction=direction,
        bins=bins,
        bin_unit=bin_unit,
        bin_continuous_input=bin_continuous_input,
        drop_na_events=drop_na_events,
        min_n_in_bin=min_n_in_bin,
    ).rename(columns={"output": "y_hat"})

    return sensitivity_by_group(
        df=df,
        groupby_col_name="unit_from_event_binned",
        confidence_interval=confidence_interval,
        stratified=stratified,
    )


def get_time_from_first_positive_to_diagnosis_df(input_df: pd.DataFrame) -> pd.DataFrame:
    """input_df must contain columns:
        y: 1 or 0
        pred: 1 or 0
        id: Patient ID
        pred_timestamps: Timestamp of prediction
        outcome_timestamps: Timestamp of outcome
    Returns a dataframe with the additional columns:
        years_from_pred_to_event: Years from first positive prediction to event
        days_from_pred_to_event: Days from first positive prediction to event
    """
    df = pl.from_pandas(input_df).with_columns(
        (pl.col("outcome_timestamps") - pl.col("pred_timestamps")).alias("time_from_pred_to_event")
    )

    ever_positives = df.filter(
        pl.col("time_from_pred_to_event").is_not_null() & pl.col("pred") == 1
    )

    plot_df = (
        ever_positives.sort("time_from_pred_to_event", descending=True)
        .groupby("id")
        .head(1)
        .with_columns(
            (pl.col("time_from_pred_to_event").dt.days() / 365.25).alias(
                "years_from_pred_to_event"
            ),
            (pl.col("time_from_pred_to_event").dt.days()).alias("days_from_pred_to_event"),
        )
    ).to_pandas()

    return plot_df
