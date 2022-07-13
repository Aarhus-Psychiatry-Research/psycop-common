"""Plotting functions for
1. AUC by calendar time
2. AUC by time from first visit
3. AUC by time until diagnosis
TODO: change default bins to something sensible in time to metric
"""
from typing import Callable, Iterable, List

import altair as alt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from wasabi import msg

from psycopt2d.utils import bin_continuous_data, round_floats_to_edge
from psycopt2d.visualization.base_charts import plot_bar_chart


def _calc_performance(df: pd.DataFrame, metric: Callable) -> float:
    """Calculates performance metrics of a df with 'y' and 'y_hat' columns.

    Args:
        df (pd.DataFrame): dataframe
        metric (Callable): which metric to calculate

    Returns:
        float: performance
    """
    if df.empty:
        return np.nan
    else:
        return metric(df["y"], df["y_hat"])


def time_difference_in_days(t1: pd.Series, t2: pd.Series) -> pd.Series:
    """Calculate time difference in days between two series.

    Args:
        t1 (pd.Series): first time series
        t2 (pd.Series): second time series

    Returns:
        pd.Series: Series of time differences in days
    """
    return (t1 - t2) / np.timedelta64(1, "D")


def create_performance_by_calendar_time_df(
    labels: Iterable[int],
    y_hat: Iterable[int],
    timestamps: Iterable[pd.Timestamp],
    metric: Callable,
    bin_period: str,
) -> pd.DataFrame:
    """Calculate performance by calendar time of prediction.

    Args:
        labels (Iterable[int]): True labels
        y_hat (Iterable[int]): Predicted label or probability depending on metric
        timestamps (Iterable[pd.Timestamp]): Timestamps of predictions
        metric (Callable): Which metric to calculate
        bin_period (str): How to bin time. "M" for year/month, "Y" for year

    Returns:
        pd.DataFrame: Dataframe ready for plotting
    """
    df = pd.DataFrame({"y": labels, "y_hat": y_hat, "timestamp": timestamps})
    df["time_bin"] = df["timestamp"].astype(f"datetime64[{bin_period}]")
    output_df = df.groupby("time_bin").apply(_calc_performance, metric)
    output_df = output_df.reset_index().rename({0: "metric"}, axis=1)
    return output_df


def plot_performance_by_calendar_time(
    labels: Iterable[int],
    y_hat: Iterable[int],
    timestamps: Iterable[pd.Timestamp],
    metric: Callable = roc_auc_score,
    bin_period: str = "M",
    y_title: str = "AUC",
) -> alt.Chart:
    """Plot performance by calendar time of prediciton.

    Args:
        labels (Iterable[int]): True labels
        y_hat (Iterable[int]): Predicted label of probability depending on metric
        timestamps (Iterable[pd.Timestamp]): Timestamps of predictions
        metric (Callable, optional): Which metric to calculate. Defaults to roc_auc_score.
        bin_period (str, optional): Which time period to bin on. Defaults to "M",
        which calculates performance on the monthly level
        y_title (str, optional): Title of y-axis. Defaults to "AUC".

    Returns:
        alt.Chart: Bar chart of performance
    """
    df = create_performance_by_calendar_time_df(
        labels=labels,
        y_hat=y_hat,
        timestamps=timestamps,
        metric=metric,
        bin_period=bin_period,
    )
    sort_order = np.arange(len(df))
    if y_title == "AUC" and metric != roc_auc_score:
        msg.warning("Title is AUC but metric might not be!")
    return plot_bar_chart(
        x_values=df["time_bin"],
        y_values=df["metric"],
        x_title="Calendar time",
        y_title=y_title,
        sort=sort_order,
    )


def create_performance_from_time_from_event_df(
    labels: Iterable[int],
    y_hat: Iterable[int],
    event_timestamps: Iterable[pd.Timestamp],
    prediction_timestamps: Iterable[pd.Timestamp],
    metric: Callable,
    direction: str,
    bins: List[int],
    pretty_bins: bool = True,
    drop_na_events: bool = False,
) -> pd.DataFrame:
    """Create dataframe for plotting performance metric from time to or from
    some event (e.g. time of diagnosis, time from first visit).

    Args:
        labels (Iterable[int]): True labels
        y_hat (Iterable[int]): Predicted probabilities or labels depending on metric
        event_timestamps (Iterable[pd.Timestamp]): Timestamp of event (e.g. first visit)
        prediction_timestamps (Iterable[pd.Timestamp]): Timestamp of prediction
        metric (Callable): Which performance metric function to use (e.g. roc_auc_score)
        direction (str, optional): Which direction to calculate time difference.
        Can either be 'prediction-event' or 'event-prediction'.
        bins (list, optional): Bins to group by.
        pretty_bins (bool, optional): Whether to prettify bin names. Defaults to True.
        drop_na_events (bool, optional): Whether to drop rows where the event is NA. Defaults to False.

    Returns:
        pd.DataFrame: Dataframe ready for plotting
    """

    df = pd.DataFrame(
        {
            "y": labels,
            "y_hat": y_hat,
            "event_timestamp": event_timestamps,
            "prediction_timestamp": prediction_timestamps,
        },
    )
    # Drop rows with no events if specified
    if drop_na_events:
        df = df.dropna(subset=["event_timestamp"])

    # Calculate difference in days between prediction and event
    if direction == "event-prediction":
        df["days_from_event"] = time_difference_in_days(
            df["event_timestamp"],
            df["prediction_timestamp"],
        )
    elif direction == "prediction-event":
        df["days_from_event"] = time_difference_in_days(
            df["prediction_timestamp"],
            df["event_timestamp"],
        )
    else:
        raise ValueError(
            f"Direction should be one of ['event-prediction', 'prediction-event'], not {direction}",
        )

    # bin data
    bin_fn = bin_continuous_data if pretty_bins else round_floats_to_edge
    df["days_from_event_binned"] = bin_fn(df["days_from_event"], bins=bins)

    # Calc performance and prettify output
    output_df = df.groupby("days_from_event_binned").apply(_calc_performance, metric)
    output_df = output_df.reset_index().rename({0: "metric"}, axis=1)
    return output_df


def plot_auc_time_from_first_visit(
    labels: Iterable[int],
    y_hat_probs: Iterable[int],
    first_visit_timestamps: Iterable[pd.Timestamp],
    prediction_timestamps: Iterable[pd.Timestamp],
    bins=[0, 1, 7, 14, 28, 182, 365, 730, 1825],
    pretty_bins: bool = True,
) -> alt.Chart:
    """Plot AUC as a function of time to first visit.

    Args:
        labels (Iterable[int]): True labels
        y_hat_probs (Iterable[int]): Predicted probabilities
        first_visit_timestamps (Iterable[pd.Timestamp]): Timestamps of the first visit
        prediction_timestamps (Iterable[pd.Timestamp]): Timestamps of the predictions
        bins (list, optional): Bins to group by. Defaults to [0, 1, 7, 14, 28, 182, 365, 730, 1825].
        pretty_bins (bool, optional): Prettify bin names. Defaults to True.

    Returns:
        alt.Chart: Altair bar chart
    """

    df = create_performance_from_time_from_event_df(
        labels=labels,
        y_hat=y_hat_probs,
        event_timestamps=first_visit_timestamps,
        prediction_timestamps=prediction_timestamps,
        direction="prediction-event",
        bins=bins,
        pretty_bins=pretty_bins,
        drop_na_events=False,
        metric=roc_auc_score,
    )

    sort_order = np.arange(len(df))
    return plot_bar_chart(
        x_values=df["days_from_event_binned"],
        y_values=df["metric"],
        x_title="Days from first visit",
        y_title="AUC",
        sort=sort_order,
    )


def plot_metric_time_until_diagnosis(
    labels: Iterable[int],
    y_hat: Iterable[int],
    diagnosis_timestamps: Iterable[pd.Timestamp],
    prediction_timestamps: Iterable[pd.Timestamp],
    bins=[
        -1825,
        -730,
        -365,
        -182,
        -28,
        -14,
        -7,
        -1,
        0,
        1,
        7,
        14,
        28,
        182,
        365,
        730,
        1825,
    ],
    pretty_bins: bool = True,
    metric: Callable = f1_score,
    y_title: str = "F1",
) -> alt.Chart:
    """Plots performance of a specified performance metric in bins of time
    until diagnosis. Rows with no date of diagnosis (i.e. no outcome) are
    removed.

    Args:
        labels (Iterable[int]): True labels
        y_hat (Iterable[int]): Predicted label
        diagnosis_timestamps (Iterable[pd.Timestamp]): Timestamp of diagnosis
        prediction_timestamps (Iterable[pd.Timestamp]): Timestamp of prediction
        y_title (str): Title for y-axis (metric name)
        bins (list, optional): Bins to group by. Negative values indicate days after
        diagnosis. Defaults to [ -1825, -730, -365, -182, -28, -14, -7, -1, 0, 1, 7, 14, 28, 182, 365, 730, 1825] (which is stupid).
        pretty_bins (bool, optional): Whether to prettify bin names. Defaults to True.
        metric (Callable): Which performance metric to use. Defaults to f1_score
    Returns:
        alt.Chart: Altair bar chart
    """
    if y_title == "F1" and metric != f1_score:
        msg.warning("Title is F1 but metric might not be!")
    df = create_performance_from_time_from_event_df(
        labels=labels,
        y_hat=y_hat,
        event_timestamps=diagnosis_timestamps,
        prediction_timestamps=prediction_timestamps,
        direction="event-prediction",
        bins=bins,
        pretty_bins=pretty_bins,
        drop_na_events=True,
        metric=metric,
    )
    sort_order = np.arange(len(df))

    return plot_bar_chart(
        x_values=df["days_from_event_binned"],
        y_values=df["metric"],
        x_title="Days to diagnosis",
        y_title=y_title,
        sort=sort_order,
    )
