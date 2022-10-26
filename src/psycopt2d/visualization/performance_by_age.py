"""Plotting function for performance by age at time of predictio."""
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional, Union

from sklearn.metrics import roc_auc_score

from psycopt2d.visualization.base_charts import plot_basic_chart
from psycopt2d.visualization.utils import create_performance_by_input


def plot_performance_by_age(
    labels: Sequence[int],
    y_hat: Sequence[int, float],
    age: Sequence[int, float],
    save_path: Optional[Path] = None,
    bins: Sequence[int, float] = [18, 25, 35, 50, 70],
    pretty_bins: Optional[bool] = True,
    metric_fn: Callable = roc_auc_score,
) -> Union[None, Path]:
    """Plot bar plot of performance (default AUC) by age at time of prediction.

    Args:
        labels (Sequence[int]): True labels
        y_hat (Sequence[int]): Predicted label or probability depending on metric
        age (Sequence[int, float]): Age at time of prediction
        bins (Sequence[int, float]): Bins to group by. Defaults to [18, 25, 35, 50, 70, 100].
        pretty_bins (bool, optional): Whether to prettify bin names. I.e. make
            bins look like "18-25" instead of "[18-25])". Defaults to True.
        metric_fn (Callable): Callable which returns the metric to calculate
        save_path (Path, optional): Path to save figure. Defaults to None.

    Returns:
        Union[None, Path]: Path to saved figure or None if not saved.
    """

    df = create_performance_by_input(
        labels=labels,
        y_hat=y_hat,
        input=age,
        input_name="age",
        metric_fn=metric_fn,
        bins=bins,
        pretty_bins=pretty_bins,
    )

    sort_order = sorted(df["age_binned"].unique())
    return plot_basic_chart(
        x_values=df["age_binned"],
        y_values=df["metric"],
        x_title="Number of HbA1c measurements",
        y_title="AUC",
        sort_x=sort_order,
        plot_type=["bar"],
        save_path=save_path,
    )
