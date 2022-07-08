from typing import Iterable

import pandas as pd

from psycopt2d.utils import difference_in_days, round_floats_to_edge
from psycopt2d.visualization.base_charts import plot_bar_chart


def create_sensitivity_by_time_to_outcome_df(
    label: Iterable[int],
    y_hat_probs: Iterable[int],
    threshold: float,
    outcome_timestamps: Iterable[pd.Timestamp],
    prediction_timestamps: Iterable[pd.Timestamp],
    bins=[0, 1, 7, 14, 28, 182, 365, 730, 1825],
) -> pd.DataFrame:
    """Calculate sensitivity by time to outcome.

    Args:
        label (Iterable[int]): Label of the data.
        y_hat (Iterable[int]): Predicted label.
        outcome_timestamps (Iterable[pd.Timestamp]): Timestamp of the outcome, if any.
        prediction_timestamps (Iterable[pd.Timestamp]): Timestamp of the prediction.
        bins (list, optional): Default bins for time to outcome. Defaults to [0, 1, 7, 14, 28, 182, 365, 730, 1825].

    Returns:
        pd.DataFrame
    """

    # Modify pandas series to 1 if y_hat is larger than threshold, otherwise 0
    y_hat = pd.Series(y_hat_probs).apply(lambda x: 1 if x > threshold else 0)

    df = pd.DataFrame(
        {
            "y": label,
            "y_hat": y_hat,
            "outcome_timestamp": outcome_timestamps,
            "prediction_timestamp": prediction_timestamps,
        },
    )

    # Convert all timestamp columns to datetime64[ns]
    df["outcome_timestamp"] = df["outcome_timestamp"].astype("datetime64[ns]")
    df["prediction_timestamp"] = df["prediction_timestamp"].astype("datetime64[ns]")

    df = df[df["y"] == 1]

    # Calculate difference in days between columns
    df["days_to_outcome"] = difference_in_days(
        df["outcome_timestamp"],
        df["prediction_timestamp"],
    )

    true = df["y"]
    pred = df["y_hat"]

    df["true_positive"] = (true == 1) & (pred == 1)
    df["false_negative"] = (true == 1) & (pred == 0)

    df["days_to_outcome_binned"] = round_floats_to_edge(
        df["days_to_outcome"],
        bins=bins,
    )

    output_df = (
        df[["days_to_outcome_binned", "true_positive", "false_negative"]]
        .groupby("days_to_outcome_binned")
        .sum()
    )

    output_df["sens"] = output_df["true_positive"] / (
        output_df["true_positive"] + output_df["false_negative"]
    )

    return output_df.reset_index()


def plot_sensitivity_by_time_to_outcome(
    labels: Iterable[int],
    y_hat_probs: Iterable[int],
    threshold: float,
    outcome_timestamps: Iterable[pd.Timestamp],
    prediction_timestamps: Iterable[pd.Timestamp],
    bins=[0, 1, 7, 14, 28, 182, 365, 730, 1825],
):
    """Plot sensitivity by time to outcome.

    Args:
        label (Iterable[int]): Label of the data.
        y_hat_probs (Iterable[int]): Predicted probability of class 1.
        threshold (float): Threshold for class 1.
        outcome_timestamps (Iterable[pd.Timestamp]): Timestamp of the outcome, if any.
        prediction_timestamps (Iterable[pd.Timestamp]): Timestamp of the prediction.
        bins (list, optional): Default bins for time to outcome. Defaults to [0, 1, 7, 14, 28, 182, 365, 730, 1825].

    Returns:
        pd.DataFrame
    """

    df = create_sensitivity_by_time_to_outcome_df(
        label=labels,
        y_hat_probs=y_hat_probs,
        threshold=threshold,
        outcome_timestamps=outcome_timestamps,
        prediction_timestamps=prediction_timestamps,
        bins=bins,
    )

    return plot_bar_chart(
        x_values=df["days_to_outcome_binned"],
        y_values=df["sens"],
        x_title="Days to outcome",
        y_title="Sensitivity",
    )
