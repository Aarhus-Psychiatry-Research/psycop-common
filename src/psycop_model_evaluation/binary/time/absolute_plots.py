from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from psycop_model_evaluation.base_charts import (
    plot_basic_chart,
)
from psycop_model_evaluation.binary.time.absolute_data import (
    create_roc_auc_by_absolute_time_df,
)
from psycop_model_evaluation.binary.time.timedelta_data import (
    create_sensitivity_by_time_to_outcome_df,
)
from psycop_model_training.model_eval.dataclasses import EvalDataset


def plot_metric_by_absolute_time(
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
    df = create_roc_auc_by_absolute_time_df(
        labels=eval_dataset.y,
        y_hat=eval_dataset.y_hat_probs,
        timestamps=eval_dataset.pred_timestamps,
        bin_period=bin_period,
    )
    sort_order = list(range(len(df)))

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
        sort_x=sort_order,  # type: ignore
        y_limits=y_limits,
        bar_count_values=df["n_in_bin"],
        bar_count_y_axis_title="Number of visits",
        plot_type=["line", "scatter"],
        save_path=save_path,
    )


def plot_recall_by_absolute_time(
    eval_dataset: EvalDataset,
    positive_rates: Union[float, Iterable[float]],
    bins: Sequence[float],
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


def plot_prob_over_time(
    timestamp: Iterable[datetime],
    pred_prob: Iterable[float],
    label: Iterable[Union[int, str]],
    outcome_timestamp: Iterable[Union[datetime, None]],
    patient_id: Iterable[Union[int, str]],
    x_axis: str = "Time from outcome",
    y_axis: str = "Model Predictive Probability",
    legend: str = "Highest Predictive Probability",
    look_behind_distance: Optional[int] = None,
    line_opacity: Optional[float] = 0.3,
    fig_size: Optional[tuple] = (10, 10),
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plot probabilities over time for a given outcome. Each element passed
    (e.g. timestamp, pred_prob etc.) must have the same length, and for each
    iterable, the i'th item must correspond to the same patient.
    Args:
        timestamp (Iterable[datetime]): Timestamps for each prediction time.
        pred_prob (Iterable[float]): The predictive probabilities of the model for each prediction time.
        label (Iterable[Union[int, str]]): True labels for each prediction time.
        outcome_timestamp (Iterable[Union[datetime, None]]): Timestamp of the
            positive outcome.
        patient_id (Iterable[Union[int, str]]): Patient ID for each prediction time. Used for
            connecting timestamp/pred-prob points into one line pr. patient.
        x_axis (str, optional): Label on x-axis. Defaults to "Time from outcome".
        y_axis (str, optional): Label of y-axis. Defaults to "Model Predictive
            Probability".
        legend (str, optional): Label on legend. Defaults to "Highest Predictive
            Probability".
        look_behind_distance (Optional[int], optional): Look-behind window. Used for
            shading the corresponding area. Defaults to None in which case no shaded
            areas is plotted.
        line_opacity (float, optional): Opacity of the line. Defaults to 0.3.
        fig_size (Optional[tuple], optional): figure size. Defaults to None.
        save_path (Optional[Path], optional): path to save figure. Defaults to None.
    Returns:
        Union[None, Path]: None if save_path is None, else path to saved figure
    Examples:
        >>> from pathlib import Path
        >>> repo_path = Path(__file__).parent.parent.parent.parent
        >>> path = repo_path / "tests" / "test_data" / "synth_eval_data.csv"
        >>> df = pd.read_csv(path)
        >>> plot_prob_over_time(
        >>>     patient_id=df["dw_ek_borger"],
        >>>     timestamp=df["timestamp"],
        >>>     pred_prob=df["pred_prob"],
        >>>     outcome_timestamp=df["timestamp_t2d_diag"],
        >>>     label=df["label"],
        >>>     look_behind=500,
        >>> )
    """

    # construct pandas df ensuring types
    plot_df = pd.DataFrame(
        {
            "timestamp": list(timestamp),
            "pred_prob": list(pred_prob),
            "outcome_timestamp": list(outcome_timestamp),
            "patient_id": list(patient_id),
            "label": list(label),
        },
    )
    # remove individuals with no outcome
    plot_df = plot_df.dropna()

    time_cols = ["timestamp", "outcome_timestamp"]
    plot_df.loc[:, time_cols] = plot_df[time_cols].apply(pd.to_datetime)
    plot_df["delta_time"] = plot_df["timestamp"] - plot_df["outcome_timestamp"]
    plot_df["delta_time"] = plot_df["delta_time"].dt.days

    plot_df["patient_id"] = plot_df["patient_id"].astype(str)
    plot_df["label"] = plot_df["label"].astype(str)

    max_pred_prob = plot_df.groupby(["patient_id"])["pred_prob"].max()
    plot_df["color"] = [max_pred_prob[id_] for id_ in plot_df["patient_id"]]

    plt.figure(figsize=fig_size)

    sns.lineplot(
        data=plot_df,
        x="delta_time",
        y="pred_prob",
        alpha=line_opacity,
        hue="color",
        palette="magma",
        legend="auto",
    )

    # Reformat y-axis values to percentage
    plt.gca().yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))  # type: ignore

    plt.xlabel(x_axis, size=14)
    plt.ylabel(y_axis, size=14)

    plt.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.gca().set_facecolor("#f2f2f2")

    plt.legend(title=legend, loc="lower right", fontsize=12)

    # Add shaded area for look-behind window
    if look_behind_distance is not None:
        plt.axvspan(
            -look_behind_distance,
            0,
            color="grey",
            alpha=0.2,
        )
        plt.text(
            -look_behind_distance / 2,
            plot_df["pred_prob"].max(),
            "Predictive window",
            horizontalalignment="center",
            verticalalignment="center",
            rotation=0,
            size=12,
        )

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()
    return save_path
