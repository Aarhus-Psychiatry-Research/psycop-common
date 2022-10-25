"""Plotting function for AUC by number of HbA1c measurements bar plot
"""
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from psycopt2d.visualization.utils import calc_performance
from psycopt2d.visualization.base_charts import plot_basic_chart


def create_performance_by_n_hba1c(
    labels: Iterable[int],
    y_hat: Iterable[int],
    n_hba1c: Iterable[int],
    metric_fn: Callable,
) -> pd.DataFrame:
    """Calculate performance by number of HbA1c measurements.

    Args:
        labels (Iterable[int]): True labels
        y_hat (Iterable[int]): Predicted label or probability depending on metric
        n_hba1c (Iterable[int]): Number of HbA1c measurements
        metric_fn (Callable): Callable which returns the metric to calculate
        bin_period (str): How to bin time. "M" for year/month, "Y" for year

    Returns:
        pd.DataFrame: Dataframe ready for plotting
    """
    df = pd.DataFrame({"y": labels, "y_hat": y_hat, "n_hba1c": n_hba1c})

    output_df = df.groupby("n_hba1c").apply(calc_performance, metric_fn)

    output_df = output_df.reset_index().rename({0: "metric"}, axis=1)
    return output_df


def plot_performance_by_n_hba1c(
    labels: Iterable[int],
    y_hat_probs: Iterable[int],
    n_hba1c: Iterable[int],
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plot bar plot of AUC by number of HbA1c measurements.

    Args:
        labels (Iterable[int]): True labels
        y_hat_probs (Iterable[int]): Predicted probabilities
        prediction_timestamps (Iterable[pd.Timestamp]): Timestamps of the predictions
        n_hba1c (Iterable[int]): Number of HbA1c measurements
        save_path (Path, optional): Path to save figure. Defaults to None.

    Returns:
        Union[None, Path]: Path to saved figure or None if not saved.
    """

    df = create_performance_by_n_hba1c(
        labels=labels,
        y_hat=y_hat_probs,
        n_hba1c=n_hba1c,
        metric_fn=roc_auc_score,
    )

    sort_order = np.arange(n_hba1c.values)
    return plot_basic_chart(
        x_values=df["n_hba1c"],
        y_values=df["metric"],
        x_title="Number of HbA1c measurements",
        y_title="AUC",
        sort_x=sort_order,
        plot_type=["bar"],
        save_path=save_path,
    )
