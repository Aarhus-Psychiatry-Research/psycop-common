from pathlib import Path
from typing import Literal, Optional, Union

from psycop.model_evaluation.base_charts import (
    plot_basic_chart,
)
from psycop.model_evaluation.binary.time.absolute_data import (
    create_roc_auc_by_absolute_time_df,
)
from psycop.model_training.training_output.dataclasses import EvalDataset


def plot_metric_by_absolute_time(
    eval_dataset: EvalDataset,
    y_title: str = "AUC",
    bin_period: Literal["H", "D", "W", "M", "Q", "Y"] = "Y",
    confidence_interval: Optional[float] = 0.95,
    pred_type: Optional[str] = "visits",
    save_path: Optional[Union[str, Path]] = None,
    y_limits: Optional[tuple[float, float]] = (0.5, 1.0),
) -> Union[None, Path]:
    """Plot performance by calendar time of prediciton.
    Args:
        eval_dataset: EvalDataset object
        y_title: Title of y-axis. Defaults to "AUC".
        bin_period: Which time period to bin on. Takes "M" for month, "Q" for quarter or "Y" for year
        pred_type: What description of prediction type to use for plotting, e.g. "number of visits".
        save_path: Path to save figure. Defaults to None.
        confidence_interval: Confidence interval  width for the performance metric. Defaults to 0.95.
        y_limits: Limits of y-axis. Defaults to (0.5, 1.0).

    Returns:
        Union[None, Path]: Path to saved figure or None if not saved.
    """
    df = create_roc_auc_by_absolute_time_df(
        labels=eval_dataset.y,
        y_hat=eval_dataset.y_hat_probs,
        timestamps=eval_dataset.pred_timestamps,
        bin_period=bin_period,
        confidence_interval=confidence_interval,
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
    ci = df["ci"].tolist() if confidence_interval else None

    return plot_basic_chart(
        x_values=df["time_bin"],
        y_values=df["metric"],
        x_title=x_titles[bin_period],
        y_title=y_title,
        sort_x=sort_order,
        y_limits=y_limits,
        confidence_interval=ci,
        bar_count_values=df["n_in_bin"],
        bar_count_y_axis_title=f"Number of {pred_type}",
        plot_type=["line", "scatter"],
        save_path=save_path,
    )
