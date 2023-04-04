from pathlib import Path
from typing import Optional, Union

from psycop_model_evaluation.base_charts import (
    plot_basic_chart,
)
from psycop_model_evaluation.binary.time.periodic_data import (
    roc_auc_by_periodic_time_df,
)
from psycop_model_training.training_output.dataclasses import EvalDataset


def plot_roc_auc_by_periodic_time(
    eval_dataset: EvalDataset,
    y_title: str = "AUC",
    bin_period: str = "Y",
    save_path: Optional[Union[str, Path]] = None,
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
    df = roc_auc_by_periodic_time_df(
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
