from collections.abc import Callable
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.metrics import roc_auc_score


def calc_performance(
    df: pd.DataFrame,
    metric: Callable,
    confidence_interval: Optional[float] = None,
    **kwargs: Any,
) -> pd.Series:
    """Calculates performance metrics of a df with 'y' and 'input_to_fn' columns.

    Args:
        df: dataframe
        metric: which metric to calculate
        confidence_interval: Confidence interval width for the performance metric. Defaults to None,
            in which case the no confidence interval is calculated.
        **kwargs: additional arguments to pass to the bootstrap function for calculating
            the confidence interval.

    Returns:
        performance
    """
    if df.empty:
        return pd.Series({"metric": np.nan, "n_in_bin": np.nan})

    if metric is roc_auc_score and len(df["y"].unique()) == 1:
        # msg.info("Only 1 class present in bin. AUC undefined. Returning np.nan") This was hit almost once per month, making it very hard to read.
        # Many of our models probably try to predict the majority class.
        # I'm not sure how exactly we want to handle this, but thousands of msg.info is not ideal.
        # For now, suppressing this message.
        return pd.Series({"metric": np.nan, "n_in_bin": np.nan})

    perf_metric = metric(df["y"], df["y_hat"])

    # If any value in n_in_bin is smaller than 5, write NaN
    n_in_bin = np.nan if len(df) < 5 else len(df)

    if confidence_interval:
        # reasonably fast and accurate defaults
        _kwargs = {
            "method": "basic",
            "n_resamples": 1000,
        }
        _kwargs.update(kwargs)

        # Calculate the confidence interval
        def metric_wrapper(
            true: np.ndarray,
            pred: np.ndarray,
            **kwargs: Any,  # noqa: ARG001
        ) -> float:
            # bootstrap function requires the metric function to
            # be able to take additional arguments (notably the length of the array)
            return metric(true, pred)

        boot = bootstrap(
            (df["y"], df["y_hat"]),
            statistic=metric_wrapper,
            confidence_level=confidence_interval,
            paired=True,
            **_kwargs,
        )
        low, high = boot.confidence_interval.low, boot.confidence_interval.high
        return pd.Series(
            {"metric": perf_metric, "n_in_bin": n_in_bin, "ci": (low, high)},
        )
    return pd.Series({"metric": perf_metric, "n_in_bin": n_in_bin})


def get_top_fraction(df: pd.DataFrame, col_name: str, fraction: float) -> pd.DataFrame:
    """
    Returns the top N percent of the data sorted by column y in a dataframe df.
    """
    # Calculate the number of rows to select
    num_rows = int(len(df) * fraction)

    # Sort the dataframe by column y and select the top N percent of rows
    sorted_df = df.sort_values(col_name, ascending=False)
    top_fraction = sorted_df.head(num_rows)

    return top_fraction
