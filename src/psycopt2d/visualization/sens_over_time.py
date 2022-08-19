from functools import partial
from typing import Iterable, List

import altair as alt
import numpy as np
import pandas as pd

from psycopt2d.utils import round_floats_to_edge


def create_sensitivity_by_time_to_outcome_df(
    label: Iterable[int],
    y_hat_probs: Iterable[int],
    positive_rate_threshold: float,
    positive_rate: float,
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
    y_hat = pd.Series(y_hat_probs).apply(
        lambda x: 1 if x > positive_rate_threshold else 0,
    )

    df = pd.DataFrame(
        {
            "y": label,
            "y_hat": y_hat,
            "outcome_timestamp": outcome_timestamps,
            "prediction_timestamp": prediction_timestamps,
        },
    )

    df = df[df["y"] == 1]

    # Calculate difference in days between columns
    df["days_to_outcome"] = (
        df["outcome_timestamp"] - df["prediction_timestamp"]
    ) / np.timedelta64(1, "D")

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
    output_df["threshold"] = positive_rate_threshold
    output_df["threshold_percentile"] = (
        positive_rate if positive_rate > 1 else positive_rate * 100
    )

    output_df = output_df.reset_index()

    # Convert days_to_outcome_binned to string for plotting
    output_df["days_to_outcome_binned"] = output_df["days_to_outcome_binned"].astype(
        str,
    )

    return output_df


def plot_sensitivity_by_time_to_outcome(
    labels: Iterable[int],
    y_hat_probs: Iterable[int],
    positive_rates: Iterable[float],
    pred_proba_thresholds: Iterable[float],
    outcome_timestamps: Iterable[pd.Timestamp],
    prediction_timestamps: Iterable[pd.Timestamp],
    bins: List[int] = [0, 28, 182, 365, 730, 1825],
):
    """Plot sensitivity by time to outcome.

    Args:
        label (Iterable[int]): Label of the data.
        y_hat_probs (Iterable[int]): Predicted probability of class 1.
        threshold_percentiles (Iterable[float]): List of thresholds to plot.
        pred_proba_thresholds (Iterable[float]): List of pred_proba thresholds above which predictions are classified as positive.
        outcome_timestamps (Iterable[pd.Timestamp]): Timestamp of the outcome, if any.
        prediction_timestamps (Iterable[pd.Timestamp]): Timestamp of the prediction.
        bins (list, optional): Default bins for time to outcome. Defaults to [0, 1, 7, 14, 28, 182, 365, 730, 1825].

    Returns:
        pd.DataFrame
    """

    func = partial(
        create_sensitivity_by_time_to_outcome_df,
        label=labels,
        y_hat_probs=y_hat_probs,
        outcome_timestamps=outcome_timestamps,
        prediction_timestamps=prediction_timestamps,
        bins=bins,
    )

    df = pd.concat(
        [
            func(
                positive_rate_threshold=pred_proba_thresholds[i],
                positive_rate=positive_rates[i],
            )
            for i in range(len(positive_rates))
        ],
        axis=0,
    )

    # Base plot
    base = alt.Chart(df).encode(
        x=alt.X(
            "days_to_outcome_binned:O",
            sort=df.days_to_outcome_binned.unique().tolist(),
            title="Days to Outcome",
        ),
        y=alt.Y("positive_rate:O", title="Positive rate", sort="-y"),
    )

    # Heatmap
    heatmap = base.mark_rect().encode(
        color=alt.Color("sens:Q", title="Sensitivity"),
    )

    text = base.mark_text(baseline="middle").encode(
        text="sens:Q",
        color=alt.condition(
            alt.datum.sens > df["sens"].median(),
            alt.value("white"),
            alt.value("black"),
        ),
    )

    plt = heatmap + text

    return plt
