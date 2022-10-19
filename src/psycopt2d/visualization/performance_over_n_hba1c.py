"""Plotting function for AUC by number of HbA1c measurements bar plot
"""
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from psycopt2d.utils import bin_continuous_data, round_floats_to_edge
from psycopt2d.visualization.utils import calc_performance
from psycopt2d.visualization.base_charts import plot_basic_chart


def create_performance_by_n_hba1c(
    labels: Iterable[int],
    y_hat: Iterable[int],
    n_hba1c: Iterable[int],
    metric_fn: Callable,
    bin_period: str,
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
    df["time_bin"] = df["timestamp"].astype(f"datetime64[{bin_period}]")

    output_df = df.groupby("time_bin").apply(calc_performance, metric_fn)

    output_df = output_df.reset_index().rename({0: "metric"}, axis=1)
    return output_df
