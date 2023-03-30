"""Generate a plot of sensitivity by time to outcome."""
from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
from psycop_model_evaluationmodel_eval.dataclasses import EvalDataset
from psycop_model_evaluationutils.utils import round_floats_to_edge


def create_sensitivity_by_time_to_outcome_df(
    eval_dataset: EvalDataset,
    desired_positive_rate: float,
    outcome_timestamps: Iterable[pd.Timestamp],
    prediction_timestamps: Iterable[pd.Timestamp],
    bins: Iterable = (0, 1, 7, 14, 28, 182, 365, 730, 1825),
    bin_delta: Literal["D", "W", "M", "Q", "Y"] = "D",
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
