"""Plotting function for performance by number of HbA1c measurements."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional, Union

from sklearn.metrics import roc_auc_score

from psycop_model_training.model_eval.base_artifacts.plots.base_charts import (
    plot_basic_chart,
)
from psycop_model_training.model_eval.base_artifacts.plots.utils import (
    create_performance_by_input,
)
from psycop_model_training.model_eval.dataclasses import EvalDataset


def plot_performance_by_n_hba1c(
    eval_dataset: EvalDataset,
    bins: Sequence[Union[int, float]] = (0, 1, 2, 5, 10),
    prettify_bins: Optional[bool] = True,
    metric_fn: Callable = roc_auc_score,
    y_limits: Optional[tuple[float, float]] = (0.5, 1.0),
    save_path: Optional[Path] = None,
    n_hba1c_col_name: Optional[str] = "eval_hba1c_within_9999_days_count_fallback_nan",
) -> Union[None, Path]:
    """Plot bar plot of performance (default AUC) by number of HbA1c
    measurements.

    Args:
        eval_dataset: EvalDataset object
        bins (Sequence[Union[int, float]]): Bins to group by. Defaults to (0, 1, 2, 5, 10, 100).
        prettify_bins (bool, optional): Whether to prettify bin names. I.e. make
            bins look like "1-7" instead of "[1-7)". Defaults to True.
        metric_fn (Callable): Callable which returns the metric to calculate
        y_limits (tuple[float, float]): y-axis limits. Defaults to (0.5, 1.0).
        save_path (Path, optional): Path to save figure. Defaults to None.
        n_hba1c_col_name (str, optional): Name of column containing number of
            HbA1c measurements. Defaults to "n_hba1c".

    Returns:
        Union[None, Path]: Path to saved figure or None if not saved.
    """

    df = create_performance_by_input(
        eval_dataset=eval_dataset,
        input=eval_dataset.custom_columns[n_hba1c_col_name],
        input_name="n_hba1c",
        metric_fn=metric_fn,
        bins=bins,
        prettify_bins=prettify_bins,
    )

    sort_order = sorted(df["n_hba1c_binned"].unique())

    return plot_basic_chart(
        x_values=df["n_hba1c_binned"],
        y_values=df["metric"],
        x_title="Number of HbA1c measurements",
        y_title="AUC",
        sort_x=sort_order,
        y_limits=y_limits,
        plot_type=["bar"],
        save_path=save_path,
    )
