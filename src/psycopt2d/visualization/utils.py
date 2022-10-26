# pylint: skip-file
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Optional

import wandb
from wandb.sdk.wandb_run import Run as wandb_run

import numpy as np
import pandas as pd

from psycopt2d.utils import bin_continuous_data
from sklearn.metrics import roc_auc_score


def log_image_to_wandb(chart_path: Path, chart_name: str, run: wandb_run):
    """Helper to log image to wandb.

    Args:
        chart_path (Path): Path to the image
        chart_name (str): Name of the chart
        run (wandb.run): Wandb run object
    """
    run.log({f"image_{chart_name}": wandb.Image(str(chart_path))})


def calc_performance(df: pd.DataFrame, metric: Callable) -> float:
    """Calculates performance metrics of a df with 'y' and 'y_hat' columns.

    Args:
        df (pd.DataFrame): dataframe
        metric (Callable): which metric to calculate

    Returns:
        float: performance
    """
    if df.empty:
        return np.nan
    elif metric is roc_auc_score and len(df["y"].unique()) == 1:
        # msg.info("Only 1 class present in bin. AUC undefined. Returning np.nan") This was hit almost once per month, making it very hard to read.
        # Many of our models probably try to predict the majority class.
        # I'm not sure how exactly we want to handle this, but thousands of msg.info is not ideal.
        # For now, suppressing this message.
        return np.nan
    else:
        return metric(df["y"], df["y_hat"])


def create_performance_by_input(
    labels: Iterable[int],
    y_hat: Iterable[int, float],
    input: Iterable[int, float],
    input_name: str,
    bins: tuple = (0, 1, 2, 5, 10, 100),
    pretty_bins: Optional[bool] = True,
    metric_fn: Callable = roc_auc_score,
) -> pd.DataFrame:
    """Calculate performance by given input values, e.g. age or number of Hbac1 measurements.

    Args:
        labels (Iterable[int]): True labels
        y_hat (Iterable[int]): Predicted label or probability depending on metric
        input (Iterable[int, float]): Input values to calculate performance by
        input_name (str): Name of the input
        bins (Iterable[float]): Bins to group by. Defaults to (0, 1, 2, 5, 10, 100).
        pretty_bins (bool, optional): Whether to prettify bin names. I.e. make
            bins look like "1-7" instead of "[1-7)". Defaults to True.
        metric_fn (Callable): Callable which returns the metric to calculate

    Returns:
        pd.DataFrame: Dataframe ready for plotting
    """
    df = pd.DataFrame({"y": labels, "y_hat": y_hat, input_name: input})

    # bin data
    if pretty_bins:
        df[f"{input_name}_binned"] = bin_continuous_data(df[input_name], bins=bins)

        output_df = df.groupby(f"{input_name}_binned").apply(
            calc_performance, metric_fn
        )

    else:
        output_df = df.groupby(input_name).apply(calc_performance, metric_fn)

    output_df = output_df.reset_index().rename({0: "metric"}, axis=1)
    return output_df
