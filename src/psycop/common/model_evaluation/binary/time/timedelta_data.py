from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from pandas import Series
from psycop.common.model_evaluation.binary.utils import (
    auroc_by_group,
    auroc_within_group,
    sensitivity_within_group,
)
from psycop.common.model_evaluation.utils import (
    bin_continuous_data,
    round_floats_to_edge,
)
from psycop.common.model_training.training_output.dataclasses import EvalDataset

TIMEDELTA_STRINGS = Literal["h", "D", "M", "Q", "Y"]


def get_timedelta_series(
    direction: Literal["t1-t2", "t2-t1"],
    bin_unit: TIMEDELTA_STRINGS,
    df: pd.DataFrame,
    t2_col_name: str,
    t1_col_name: str,
) -> pd.Series:
    """Calculate the time difference between two timestamps."""
    if direction == "t1-t2":
        df["unit_from_event"] = (df[t1_col_name] - df[t2_col_name]) / np.timedelta64(
            1,
            bin_unit,
        )  # type: ignore
    elif direction == "t2-t1":
        df["unit_from_event"] = (df[t2_col_name] - df[t1_col_name]) / np.timedelta64(
            1,
            bin_unit,
        )  # type: ignore
    else:
        raise ValueError(
            f"Direction should be one of ['t1-t2', 't2-t1'], not {direction}",
        )

    return df["unit_from_event"]


def get_timedelta_df(
    y: Iterable[int],
    y_hat: Iterable[float],
    time_one: Iterable[pd.Timestamp],
    time_two: Iterable[pd.Timestamp],
    direction: Literal["t1-t2", "t2-t1"],
    bins: Sequence[float],
    bin_unit: TIMEDELTA_STRINGS,
    bin_continuous_input: bool = True,
    drop_na_events: bool = True,
    min_n_in_bin: int = 5,
) -> pd.DataFrame:
    """Get a timedelta dataframe with the time difference between two timestamps."""
    df = pd.DataFrame(
        {
            "y": y,
            "y_hat": y_hat,
            "t1_timestamp": time_one,
            "t2_timestamp": time_two,
        },
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
            df["unit_from_event"],
            bins=bins,
            min_n_in_bin=min_n_in_bin,
        )
    else:
        df["unit_from_event_binned"] = round_floats_to_edge(
            df["unit_from_event"],
            bins=bins,
        )

    return df


def get_auroc_by_timedelta_df(
    y: Iterable[int],
    y_pred_proba: Iterable[float],
    time_one: Iterable[pd.Timestamp],
    time_two: Iterable[pd.Timestamp],
    direction: Literal["t1-t2", "t2-t1"],
    bins: Sequence[float],
    bin_unit: TIMEDELTA_STRINGS,
    confidence_interval: bool = True,
    bin_continuous_input: bool = True,
    drop_na_events: bool = True,
    min_n_in_bin: int = 5,
) -> pd.DataFrame:
    """Create dataframe for plotting performance metric from time to or from
    some event (e.g. time of diagnosis, time from first visit).

    Args:
        y: True labels
        y_pred_proba: The predicted probabilities by the function.
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
        min_n_in_bin: Minimum number of rows in a bin to include in output. Defaults to 10.
    Returns:
        pd.DataFrame: Dataframe ready for plotting where each row represents a bin.
    """
    df = get_timedelta_df(
        y=y,
        y_hat=y_pred_proba,
        time_one=time_one,
        time_two=time_two,
        direction=direction,
        bins=bins,
        bin_unit=bin_unit,
        bin_continuous_input=bin_continuous_input,
        drop_na_events=drop_na_events,
        min_n_in_bin=min_n_in_bin,
    )

    return auroc_by_group(
        df=df,
        groupby_col_name="unit_from_event_binned",
        confidence_interval=confidence_interval,
    )


def get_sensitivity_by_timedelta_df(
    y: Iterable[int],
    y_pred: Iterable[float],
    time_one: Iterable[pd.Timestamp],
    time_two: Iterable[pd.Timestamp],
    direction: Literal["t1-t2", "t2-t1"],
    bins: Sequence[float],
    bin_unit: TIMEDELTA_STRINGS,
    confidence_interval: bool = True,
    bin_continuous_input: bool = True,
    drop_na_events: bool = True,
    min_n_in_bin: int = 5,
) -> pd.DataFrame:
    """Create dataframe for plotting performance metric from time to or from
    some event (e.g. time of diagnosis, time from first visit).

    Args:
        y: True labels
        y_pred: The predicted class.
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
        min_n_in_bin: Minimum number of rows in a bin to include in output. Defaults to 10.
    Returns:
        pd.DataFrame: Dataframe ready for plotting where each row represents a bin.
    """
    df = get_timedelta_df(
        y=y,
        y_hat=y_pred,
        time_one=time_one,
        time_two=time_two,
        direction=direction,
        bins=bins,
        bin_unit=bin_unit,
        bin_continuous_input=bin_continuous_input,
        drop_na_events=drop_na_events,
        min_n_in_bin=min_n_in_bin,
    )

    return df.groupby(["unit_from_event_binned"], as_index=False).apply(
        sensitivity_within_group,  # type: ignore
        y_true=df["y"],
        y_pred=df["y_hat"],
        confidence_interval=confidence_interval,
    )


def create_sensitivity_by_time_to_outcome_df(
    eval_dataset: EvalDataset,
    desired_positive_rate: float,
    outcome_timestamps: Series,
    prediction_timestamps: Series,
    bins: Sequence[float] = (0, 1, 7, 14, 28, 182, 365, 730, 1825),
    bin_delta: TIMEDELTA_STRINGS = "D",
) -> pd.DataFrame:
    """Calculate sensitivity by time to outcome.
    Args:
        eval_dataset (EvalDataset): Eval dataset.
        desired_positive_rate (float): Takes the top positive_rate% of predicted probabilities and turns them into 1, the rest 0.
        outcome_timestamps (Iterable[pd.Timestamp]): Timestamp of the outcome, if any.
        prediction_timestamps (Iterable[pd.Timestamp]): Timestamp of the prediction.
        bins (list, optional): Default bins for time to outcome. Defaults to [0, 1, 7, 14, 28, 182, 365, 730, 1825].
        bin_delta (str, optional): The unit of time for the bins. Defaults to "D".
        n_bootstraps (int, optional): Number of bootstraps to use for confidence intervals. Defaults to 1000.
    Returns:
        pd.DataFrame
    """

    y_hat_series, actual_positive_rate = eval_dataset.get_predictions_for_positive_rate(
        desired_positive_rate=desired_positive_rate,
    )

    df = pd.DataFrame(
        {
            "y": eval_dataset.y,
            "y_hat": y_hat_series,
            "outcome_timestamp": outcome_timestamps,
            "prediction_timestamp": prediction_timestamps,
        },
    )

    df = df[df["y"] == 1]

    df = get_timedelta_df(
        y=df["y"],
        y_hat=df["y_hat"],
        time_one=df["prediction_timestamp"],
        time_two=df["outcome_timestamp"],
        direction="t2-t1",
        bins=bins,
        bin_unit=bin_delta,
        bin_continuous_input=True,
        drop_na_events=True,
        min_n_in_bin=5,
    )
    df = df.rename(columns={"unit_from_event_binned": "days_to_outcome_binned"})

    df["true_positive"] = (df["y"] == 1) & (df["y_hat"] == 1)
    df["false_negative"] = (df["y"] == 1) & (df["y_hat"] == 0)

    df_with_metric = (
        df.groupby("days_to_outcome_binned")
        .apply(
            func=sensitivity_within_group,  # type: ignore
            y_true=df["y"],
            y_pred=df["y_hat"],
            confidence_interval=True,
        )
        .reset_index()
    )

    # Super hacky! Even though the input dfs have a specified shape,
    # we sometimes need to pivot. Instead of debugging now, we'll just
    # hack it.
    if "level_1" in df_with_metric.columns:
        output_df = df_with_metric.pivot(
            index="days_to_outcome_binned",
            columns="level_1",
            values=0,
        )
    else:
        output_df = df_with_metric

    # Get proportion of y_hat == 1, which is equal to the actual positive rate in the data.
    threshold_percentile = round(
        actual_positive_rate * 100,
        2,
    )
    output_df["sens"] = output_df["sensitivity"]
    output_df = output_df.drop("sensitivity", axis=1)

    # Prep for plotting
    ## Save the threshold for each bin
    output_df["desired_positive_rate"] = desired_positive_rate
    output_df["threshold_percentile"] = threshold_percentile
    output_df["actual_positive_rate"] = round(actual_positive_rate, 2)

    output_df = output_df.dropna(subset=["n_in_bin"])
    output_df = output_df.reset_index()

    df["days_to_outcome_binned"] = pd.Categorical(
        df["days_to_outcome_binned"],
        categories=df["days_to_outcome_binned"].index.tolist(),
        ordered=True,
    )

    return output_df
