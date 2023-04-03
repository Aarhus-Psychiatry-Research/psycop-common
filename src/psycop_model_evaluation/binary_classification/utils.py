from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def calc_performance(df: pd.DataFrame, metric: Callable) -> pd.Series:
    """Calculates performance metrics of a df with 'y' and 'input_to_fn' columns.

    Args:
        df (pd.DataFrame): dataframe
        metric (Callable): which metric to calculate

    Returns:
        float: performance
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
