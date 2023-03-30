"""Plotting functions for
1. AUC by calendar time
2. AUC by time from first visit
3. AUC by time until diagnosis
"""

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from psycop_model_evaluationmodel_eval.base_artifacts.plots.base_charts import (
    plot_basic_chart,
)
from psycop_model_evaluationmodel_eval.base_artifacts.plots.sens_over_time import (
    create_sensitivity_by_time_to_outcome_df,
)
from psycop_model_evaluationmodel_eval.base_artifacts.plots.utils import (
    calc_performance,
)
from psycop_model_evaluationmodel_eval.dataclasses import EvalDataset
from psycop_model_evaluationutils.utils import bin_continuous_data, round_floats_to_edge
from sklearn.metrics import recall_score, roc_auc_score


def plot_recall_by_calendar_time(
    eval_dataset: EvalDataset,
    positive_rates: Union[float, Iterable[float]],
    bins: Iterable[float],
    bin_unit: Literal["H", "D", "W", "M", "Q", "Y"] = "D",
    y_title: str = "Sensitivity (Recall)",
    y_limits: Optional[tuple[float, float]] = None,
    save_path: Optional[Union[Path, str]] = None,
) -> Union[None, Path]:
    """Plot performance by calendar time of prediciton.

    Args:
        eval_dataset (EvalDataset): EvalDataset object
        positive_rates (Union[float, Iterable[float]]): Positive rates to plot. Takes the top X% of predicted probabilities and discretises them into binary predictions.
        bins (Iterable[float], optional): Bins to use for time to outcome.
        bin_unit (Literal["H", "D", "M", "Q", "Y"], optional): Unit of time to bin by. Defaults to "D".
        y_title (str): Title of y-axis. Defaults to "AUC".
        save_path (str, optional): Path to save figure. Defaults to None.
        y_limits (tuple[float, float], optional): Limits of y-axis. Defaults to (0.5, 1.0).

    Returns:
        Union[None, Path]: Path to saved figure or None if not saved.
    """
    if not isinstance(positive_rates, Iterable):
        positive_rates = [positive_rates]
    positive_rates = list(positive_rates)

    dfs = [
        create_sensitivity_by_time_to_outcome_df(
            eval_dataset=eval_dataset,
            desired_positive_rate=positive_rate,
            outcome_timestamps=eval_dataset.outcome_timestamps,
            prediction_timestamps=eval_dataset.pred_timestamps,
            bins=bins,
            bin_delta=bin_unit,
        )
        for positive_rate in positive_rates
    ]

    bin_delta_to_str = {
        "H": "Hour",
        "D": "Day",
        "W": "Week",
        "M": "Month",
        "Q": "Quarter",
        "Y": "Year",
    }

    x_title_unit = bin_delta_to_str[bin_unit]
    return plot_basic_chart(
        x_values=dfs[0]["days_to_outcome_binned"],
        y_values=[df["sens"] for df in dfs],
        x_title=f"{x_title_unit}s to event",
        labels=[df["actual_positive_rate"][0] for df in dfs],
        y_title=y_title,
        y_limits=y_limits,
        flip_x_axis=True,
        plot_type=["line", "scatter"],
        save_path=save_path,
    )


def create_roc_auc_by_calendar_time_df(
    labels: Iterable[int],
    y_hat: Iterable[float],
    timestamps: Iterable[pd.Timestamp],
    bin_period: str,
) -> pd.DataFrame:
    """Calculate performance by calendar time of prediction.

    Args:
        labels (Iterable[int]): True labels
        y_hat (Iterable[int, float]): Predicted probabilities or labels depending on metric
        timestamps (Iterable[pd.Timestamp]): Timestamps of predictions
        bin_period (str): How to bin time. Takes "M" for month, "Q" for quarter or "Y" for year

    Returns:
        pd.DataFrame: Dataframe ready for plotting
    """
    df = pd.DataFrame({"y": labels, "y_hat": y_hat, "timestamp": timestamps})

    df["time_bin"] = pd.PeriodIndex(df["timestamp"], freq=bin_period).format()

    output_df = df.groupby("time_bin").apply(
        func=calc_performance,
        metric=roc_auc_score,
    )

    output_df = output_df.reset_index().rename({0: "metric"}, axis=1)

    return output_df


def plot_metric_by_calendar_time(
    eval_dataset: EvalDataset,
    y_title: str = "AUC",
    bin_period: Literal["H", "D", "W", "M", "Q", "Y"] = "Y",
    save_path: Optional[str] = None,
    y_limits: Optional[tuple[float, float]] = (0.5, 1.0),
) -> Union[None, Path]:
    """Plot performance by calendar time of prediciton.

    Args:
        eval_dataset (EvalDataset): EvalDataset object
        y_title (str): Title of y-axis. Defaults to "AUC".
        bin_period (str): Which time period to bin on. Takes "M" for month, "Q" for quarter or "Y" for year
        save_path (str, optional): Path to save figure. Defaults to None.
        y_limits (tuple[float, float], optional): Limits of y-axis. Defaults to (0.5, 1.0).

    Returns:
        Union[None, Path]: Path to saved figure or None if not saved.
    """
    df = create_roc_auc_by_calendar_time_df(
        labels=eval_dataset.y,
        y_hat=eval_dataset.y_hat_probs,
        timestamps=eval_dataset.pred_timestamps,
        bin_period=bin_period,
    )
    sort_order = np.arange(len(df))

    x_titles = {
        "H": "Hour",
        "D": "Day",
        "W": "Week",
        "M": "Month",
        "Q": "Quarter",
        "Y": "Year",
    }

    return plot_basic_chart(
        x_values=df["time_bin"],
        y_values=df["metric"],
        x_title=x_titles[bin_period],
        y_title=y_title,
        sort_x=sort_order,
        y_limits=y_limits,
        bar_count_values=df["n_in_bin"],
        bar_count_y_axis_title="Number of visits",
        plot_type=["line", "scatter"],
        save_path=save_path,
    )


def roc_auc_by_cyclic_time_df(
    labels: Iterable[int],
    y_hat: Iterable[float],
    timestamps: Iterable[pd.Timestamp],
    bin_period: str,
) -> pd.DataFrame:
    """Calculate performance by cyclic time period of prediction time data
    frame. Cyclic time periods include e.g. day of week, hour of day, etc.

    Args:
        labels (Iterable[int]): True labels
        y_hat (Iterable[int, float]): Predicted probabilities or labels depending on metric
        timestamps (Iterable[pd.Timestamp]): Timestamps of predictions
        bin_period (str): Which cyclic time period to bin on. Takes "H" for hour of day, "D" for day of week and "M" for month of year.

    Returns:
        pd.DataFrame: Dataframe ready for plotting
    """
    df = pd.DataFrame({"y": labels, "y_hat": y_hat, "timestamp": timestamps})

    if bin_period == "H":
        df["time_bin"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H")
    elif bin_period == "D":
        df["time_bin"] = pd.to_datetime(df["timestamp"]).dt.strftime("%A")
        # Sort days of week correctly
        df["time_bin"] = pd.Categorical(
            df["time_bin"],
            categories=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            ordered=True,
        )
    elif bin_period == "M":
        df["time_bin"] = pd.to_datetime(df["timestamp"]).dt.strftime("%B")
        # Sort months correctly
        df["time_bin"] = pd.Categorical(
            df["time_bin"],
            categories=[
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ],
            ordered=True,
        )
    else:
        raise ValueError(
            "bin_period must be 'H' for hour of day, 'D' for day of week or 'M' for month of year",
        )

    output_df = df.groupby("time_bin").apply(
        func=calc_performance,
        metric=roc_auc_score,
    )

    output_df = output_df.reset_index().rename({0: "metric"}, axis=1)

    return output_df


def plot_roc_auc_by_cyclic_time(
    eval_dataset: EvalDataset,
    y_title: str = "AUC",
    bin_period: str = "Y",
    save_path: Optional[str] = None,
    y_limits: Optional[tuple[float, float]] = (0.5, 1.0),
) -> Union[None, Path]:
    """Plot performance by cyclic time period of prediction time. Cyclic time
    periods include e.g. day of week, hour of day, etc.

    Args:
        eval_dataset (EvalDataset): EvalDataset object
        y_title (str): Title for y-axis (metric name). Defaults to "AUC"
        bin_period (str): Which cyclic time period to bin on. Takes "H" for hour of day, "D" for day of week and "M" for month of year.
        save_path (str, optional): Path to save figure. Defaults to None.
        y_limits (tuple[float, float], optional): Limits of y-axis. Defaults to (0.5, 1.0).

    Returns:
        Union[None, Path]: Path to saved figure or None if not saved.
    """
    df = roc_auc_by_cyclic_time_df(
        labels=eval_dataset.y,
        y_hat=eval_dataset.y_hat_probs,
        timestamps=eval_dataset.pred_timestamps,
        bin_period=bin_period,
    )

    return plot_basic_chart(
        x_values=df["time_bin"],
        y_values=df["metric"],
        x_title="Hour of day"
        if bin_period == "H"
        else "Day of week"
        if bin_period == "D"
        else "Month of year",
        y_title=y_title,
        y_limits=y_limits,
        plot_type=["line", "scatter"],
        bar_count_values=df["n_in_bin"],
        bar_count_y_axis_title="Number of visits",
        save_path=save_path,
    )


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

    # Calc performance and prettify output
    output_df = df.groupby(["unit_from_event_binned"], as_index=False).apply(
        calc_performance,
        metric=metric_fn,
    )

    return output_df


def plot_roc_auc_by_time_from_first_visit(
    eval_dataset: EvalDataset,
    bins: tuple = (0, 28, 182, 365, 730, 1825),
    bin_unit: Literal["H", "D", "M", "Q", "Y"] = "D",
    bin_continuous_input: bool = True,
    y_limits: tuple[float, float] = (0.5, 1.0),
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plot AUC as a function of time from first visit.

    Args:
        eval_dataset (EvalDataset): EvalDataset object
        bins (list, optional): Bins to group by. Defaults to [0, 28, 182, 365, 730, 1825].
        bin_unit (Literal["H", "D", "M", "Q", "Y"], optional): Unit of time to bin by. Defaults to "D".
        bin_continuous_input (bool, optional): Whether to bin input. Defaults to True.
        y_limits (tuple[float, float], optional): Limits of y-axis. Defaults to (0.5, 1.0).
        save_path (Path, optional): Path to save figure. Defaults to None.

    Returns:
        Union[None, Path]: Path to saved figure or None if not saved.
    """
    eval_df = pd.DataFrame(
        {"ids": eval_dataset.ids, "pred_timestamps": eval_dataset.pred_timestamps},
    )

    first_visit_timestamps = eval_df.groupby("ids")["pred_timestamps"].transform("min")

    df = create_performance_by_timedelta(
        y=eval_dataset.y,
        y_to_fn=eval_dataset.y_hat_probs,
        metric_fn=roc_auc_score,
        time_one=first_visit_timestamps,
        time_two=eval_dataset.pred_timestamps,
        direction="t2-t1",
        bins=list(bins),
        bin_unit=bin_unit,
        bin_continuous_input=bin_continuous_input,
        drop_na_events=False,
    )

    bin_unit2str = {
        "H": "Hours",
        "D": "Days",
        "M": "Months",
        "Q": "Quarters",
        "Y": "Years",
    }

    sort_order = np.arange(len(df))
    return plot_basic_chart(
        x_values=df["unit_from_event_binned"],
        y_values=df["metric"],
        x_title=f"{bin_unit2str[bin_unit]} from first visit",
        y_title="AUC",
        sort_x=sort_order,
        y_limits=y_limits,
        plot_type=["line", "scatter"],
        bar_count_values=df["n_in_bin"],
        bar_count_y_axis_title="Number of visits",
        save_path=save_path,
    )


def plot_sensitivity_by_time_until_diagnosis(
    eval_dataset: EvalDataset,
    bins: Sequence[int] = (
        -1825,
        -730,
        -365,
        -182,
        -28,
        -0,
    ),
    bin_unit: Literal["H", "D", "M", "Q", "Y"] = "D",
    bin_continuous_input: bool = True,
    positive_rate: float = 0.5,
    y_title: str = "Sensitivity (recall)",
    y_limits: Optional[tuple[float, float]] = None,
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plots performance of a specified performance metric in bins of time
    until diagnosis. Rows with no date of diagnosis (i.e. no outcome) are
    removed.

    Args:
        eval_dataset (EvalDataset): EvalDataset object
        bins (list, optional): Bins to group by. Negative values indicate days after
        bin_unit (Literal["H", "D", "M", "Q", "Y"], optional): Unit of time to bin by. Defaults to "D".
        diagnosis. Defaults to (-1825, -730, -365, -182, -28, -14, -7, -1, 0)
        bin_continuous_input (bool, optional): Whether to bin input. Defaults to True.
        positive_rate (float, optional): Takes the top positive_rate% of predicted probabilities and turns them into 1, the rest 0.
        y_title (str): Title for y-axis (metric name)
        y_limits (tuple[float, float], optional): Limits of y-axis. Defaults to None.
        save_path (Path, optional): Path to save figure. Defaults to None.

    Returns:
        Union[None, Path]: Path to saved figure if save_path is specified, else None
    """
    df = create_performance_by_timedelta(
        y=eval_dataset.y,
        y_to_fn=eval_dataset.get_predictions_for_positive_rate(
            desired_positive_rate=positive_rate,
        )[0],
        metric_fn=recall_score,
        time_one=eval_dataset.outcome_timestamps,
        time_two=eval_dataset.pred_timestamps,
        direction="t1-t2",
        bins=bins,
        bin_unit=bin_unit,
        bin_continuous_input=bin_continuous_input,
        min_n_in_bin=5,
        drop_na_events=True,
    )
    sort_order = np.arange(len(df))

    bin_unit2str = {
        "H": "Hours",
        "D": "Days",
        "M": "Months",
        "Q": "Quarters",
        "Y": "Years",
    }

    return plot_basic_chart(
        x_values=df["unit_from_event_binned"],
        y_values=df["metric"],
        x_title=f"{bin_unit2str[bin_unit]} to diagnosis",
        y_title=y_title,
        sort_x=sort_order,
        bar_count_values=df["n_in_bin"],
        y_limits=y_limits,
        plot_type=["scatter", "line"],
        save_path=save_path,
    )
