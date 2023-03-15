# pylint: skip-file
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import wandb
from psycop_model_training.model_eval.dataclasses import EvalDataset
from psycop_model_training.utils.utils import bin_continuous_data
from sklearn.metrics import roc_auc_score


def log_image_to_wandb(chart_path: Path, chart_name: str):
    """Helper to log image to wandb.

    Args:
        chart_path (Path): Path to the image
        chart_name (str): Name of the chart
        run (wandb.run): Wandb run object
    """
    wandb.log({f"image_{chart_name}": wandb.Image(str(chart_path))})


def calc_performance(df: pd.DataFrame, metric: Callable) -> pd.Series:
    """Calculates performance metrics of a df with 'y' and 'y_hat' columns.

    Args:
        df (pd.DataFrame): dataframe
        metric (Callable): which metric to calculate

    Returns:
        float: performance
    """
    if df.empty:
        return pd.Series({"metric": np.nan})
    if metric is roc_auc_score and len(df["y"].unique()) == 1:
        # msg.info("Only 1 class present in bin. AUC undefined. Returning np.nan") This was hit almost once per month, making it very hard to read.
        # Many of our models probably try to predict the majority class.
        # I'm not sure how exactly we want to handle this, but thousands of msg.info is not ideal.
        # For now, suppressing this message.
        return pd.Series({"metric": np.nan})

    perf_metric = metric(df["y"], df["y_hat"])

    # If any value in n_in_bin is smaller than 5, write NaN
    n_in_bin = np.nan if len(df) < 5 else len(df)

    return pd.Series({"metric": perf_metric, "n_in_bin": n_in_bin})


def metric_fn_to_input(metric_fn: Callable, eval_dataset: EvalDataset) -> str:
    """Selects the input to use for the metric function.

    Args:
        metric_fn (Callable): Metric function
        eval_dataset (EvalDataset): Evaluation dataset

    Returns:
        str: Input name
    """
    fn2input = {roc_auc_score: eval_dataset.y_hat_probs}

    if metric_fn in fn2input:
        return fn2input[metric_fn]

    raise ValueError(f"Don't know which input to use for {metric_fn}")


def create_performance_by_input(
    eval_dataset: EvalDataset,
    input_values: Sequence[float],
    input_name: str,
    bins: Sequence[float] = (0, 1, 2, 5, 10),
    bin_continuous_input: Optional[bool] = True,
    metric_fn: Callable = roc_auc_score,
) -> pd.DataFrame:
    """Calculate performance by given input values, e.g. age or number of hbac1
    measurements.bio.

    Args:
        eval_dataset: EvalDataset object
        input_values (Sequence[float]): Input values to calculate performance by
        input_name (str): Name of the input
        bins (Sequence[float]): Bins to group by. Defaults to (0, 1, 2, 5, 10, 100).
        bin_continuous_input (bool, optional): Whether to bin input. Defaults to True.
        metric_fn (Callable): Callable which returns the metric to calculate

    Returns:
        pd.DataFrame: Dataframe ready for plotting
    """
    df = pd.DataFrame(
        {
            "y": eval_dataset.y,
            "y_hat": metric_fn_to_input(metric_fn=metric_fn, eval_dataset=eval_dataset),
            input_name: input_values,
        },
    )

    # bin data and calculate metric per bin
    if bin_continuous_input:
        df[f"{input_name}_binned"], _ = bin_continuous_data(df[input_name], bins=bins)

        output_df = df.groupby(f"{input_name}_binned").apply(
            calc_performance,  # type: ignore
            metric_fn,
        )

    else:
        output_df = df.groupby(input_name).apply(calc_performance, metric_fn)  # type: ignore

    final_df = output_df.reset_index().rename({0: "metric"}, axis=1)
    return final_df
