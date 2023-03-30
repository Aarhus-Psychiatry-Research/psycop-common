from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, Optional, Union

from psycop_model_evaluation.binary_classification.plots.base_charts import (
    plot_basic_chart,
)
from psycop_model_evaluation.binary_classification.plots.robustness.time.linear.absolute.data_by_absolute_time import (
    create_roc_auc_by_linear_time_df,
)
from psycop_model_evaluation.binary_classification.plots.robustness.time.linear.timedelta.data_by_timedelta import (
    create_sensitivity_by_time_to_outcome_df,
)
from psycop_model_training.model_eval.dataclasses import EvalDataset


def plot_metric_by_linear_time(
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
    df = create_roc_auc_by_linear_time_df(
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


def plot_recall_by_linear_time(
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
