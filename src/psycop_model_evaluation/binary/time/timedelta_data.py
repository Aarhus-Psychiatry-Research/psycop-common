from collections.abc import Callable, Iterable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from pandas import Series
from psycop_model_evaluation.binary.utils import (
    calc_performance,
)
from psycop_model_evaluation.utils import (
    bin_continuous_data,
    round_floats_to_edge,
)
from psycop_model_training.training_output.dataclasses import EvalDataset


def create_performance_by_timedelta(
    y: Iterable[int],
    y_to_fn: Iterable[float],
    metric_fn: Callable,
    time_one: Iterable[pd.Timestamp],
    time_two: Iterable[pd.Timestamp],
    direction: Literal["t1-t2", "t2-t1"],
    bins: Sequence[float],
    bin_unit: Literal["H", "D", "M", "Q", "Y"],
    bin_continuous_input: bool = True,
    drop_na_events: bool = True,
    min_n_in_bin: int = 5,
) -> pd.DataFrame:
    """Create dataframe for plotting performance metric from time to or from
    some event (e.g. time of diagnosis, time from first visit).
    Args:
        y (Iterable[int]): True labels
        y_to_fn (Iterable[float]): The input to the function
        metric_fn (Callable): Function to calculate metric
        time_one (Iterable[pd.Timestamp]): Timestamps for time one (e.g. first visit).
        time_two (Iterable[pd.Timestamp]): Timestamps for time two.
        direction (str): Which direction to calculate time difference.
        Can either be 't2-t1' or 't1-t2'.
        bins (Iterable[float]): Bins to group by.
        bin_unit (Literal["H", "D", "M", "Q", "Y"]): Unit of time to use for bins.
        bin_continuous_input (bool, ): Whether to bin input. Defaults to True.
        drop_na_events (bool, ): Whether to drop rows where the event is NA. Defaults to True.
        min_n_in_bin (int, ): Minimum number of rows in a bin to include in output. Defaults to 10.
    Returns:
        pd.DataFrame: Dataframe ready for plotting where each row represents a bin.
    """
    df = pd.DataFrame(
        {
            "y": y,
            "y_hat": y_to_fn,
            "t1_timestamp": time_one,
            "t2_timestamp": time_two,
        },
    )
    # Drop rows with no events if specified
    if drop_na_events:
        df = df.dropna(subset=["t1_timestamp"])

    # Calculate difference in days between prediction and event
    if direction == "t1-t2":
        df["unit_from_event"] = (
            df["t1_timestamp"] - df["t2_timestamp"]
        ) / np.timedelta64(
            1,
            bin_unit,
        )  # type: ignore

    elif direction == "t2-t1":
        df["unit_from_event"] = (
            df["t2_timestamp"] - df["t1_timestamp"]
        ) / np.timedelta64(
            1,
            bin_unit,
        )  # type: ignore

    else:
        raise ValueError(
            f"Direction should be one of ['t1-t2', 't2-t1'], not {direction}",
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

    return df.groupby(["unit_from_event_binned"], as_index=False).apply(
        calc_performance,  # type: ignore
        metric=metric_fn,
    )


def create_sensitivity_by_time_to_outcome_df(
    eval_dataset: EvalDataset,
    desired_positive_rate: float,
    outcome_timestamps: Series,
    prediction_timestamps: Series,
    bins: Sequence[float] = (0, 1, 7, 14, 28, 182, 365, 730, 1825),
    bin_delta: Literal["H", "D", "W", "M", "Q", "Y"] = "D",
) -> pd.DataFrame:
    """Calculate sensitivity by time to outcome.
    Args:
        eval_dataset (EvalDataset): Eval dataset.
        desired_positive_rate (float): Takes the top positive_rate% of predicted probabilities and turns them into 1, the rest 0.
        outcome_timestamps (Iterable[pd.Timestamp]): Timestamp of the outcome, if any.
        prediction_timestamps (Iterable[pd.Timestamp]): Timestamp of the prediction.
        bins (list, optional): Default bins for time to outcome. Defaults to [0, 1, 7, 14, 28, 182, 365, 730, 1825].
        bin_delta (str, optional): The unit of time for the bins. Defaults to "D".
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

    # Get proportion of y_hat == 1, which is equal to the actual positive rate in the data.
    threshold_percentile = round(
        actual_positive_rate * 100,
        2,
    )

    df = df[df["y"] == 1]

    # Calculate difference in days between columns
    df["days_to_outcome"] = (
        df["outcome_timestamp"] - df["prediction_timestamp"]
    ) / np.timedelta64(
        1,
        bin_delta,
    )  # type: ignore

    df["true_positive"] = (df["y"] == 1) & (df["y_hat"] == 1)
    df["false_negative"] = (df["y"] == 1) & (df["y_hat"] == 0)

    df["days_to_outcome_binned"] = round_floats_to_edge(
        df["days_to_outcome"],
        bins=bins,
    )

    output_df = (
        df[["days_to_outcome_binned", "true_positive", "false_negative"]]
        .groupby("days_to_outcome_binned")
        .sum()
    )

    output_df["sens"] = round(
        output_df["true_positive"]
        / (output_df["true_positive"] + output_df["false_negative"]),
        2,
    )

    # Prep for plotting
    ## Save the threshold for each bin
    output_df["desired_positive_rate"] = desired_positive_rate
    output_df["threshold_percentile"] = threshold_percentile
    output_df["actual_positive_rate"] = round(actual_positive_rate, 2)

    output_df = output_df.reset_index()

    # Convert days_to_outcome_binned to string for plotting
    output_df["days_to_outcome_binned"] = output_df["days_to_outcome_binned"].astype(
        str,
    )

    return output_df
