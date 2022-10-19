# pylint: skip-file
from collections.abc import Callable
from pathlib import Path

import wandb
from wandb.sdk.wandb_run import Run as wandb_run

import numpy as np
import pandas as pd
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
