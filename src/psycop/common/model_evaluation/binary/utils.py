import numpy as np
import pandas as pd
from psycop.common.model_evaluation.binary.bootstrap_estimates import (
    bootstrap_estimates,
)
from sklearn.metrics import recall_score, roc_auc_score


def auroc_by_group(
    df: pd.DataFrame,
    confidence_interval: bool = True,
    n_bootstraps: int = 100,
) -> pd.Series:
    """Get the auroc within a dataframe."""
    if df.empty or len(df["y"].unique()) == 1 or len(df) < 5:
        # Many of our models probably try to predict the majority class.
        # I'm not sure how exactly we want to handle this, but thousands of msg.info is not ideal.
        # For now, suppressing this message.
        # Also protect against fewer than 5 in bin
        return pd.Series({"auroc": np.nan, "n_in_bin": np.nan})

    auroc = roc_auc_score(df["y"], df["y_hat"])
    auroc_by_group = {"auroc": auroc, "n_in_bin": len(df)}

    if confidence_interval:
        ci = bootstrap_estimates(
            roc_auc_score,
            n_bootstraps=n_bootstraps,
            ci_width=0.95,
            input_1=df["y"],
            input_2=df["y_hat"],
        )
        auroc_by_group["ci"] = ci

    return pd.Series(auroc_by_group)


def sensitivity_by_group(
    df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    confidence_interval: bool = True,
    n_bootstraps: int = 100,
) -> pd.Series:
    """Get the sensitivity within a dataframe."""
    if df.empty or len(df) < 5:
        # Protect against fewer than 5 in bin
        return pd.Series({"sensitivity": np.nan, "n_in_bin": np.nan})

    sensitivity = recall_score(y_true, y_pred)
    sensitivity_by_group = {"sensitivity": sensitivity, "n_in_bin": len(df)}

    if confidence_interval:
        ci = bootstrap_estimates(
            recall_score,
            n_bootstraps=n_bootstraps,
            ci_width=0.95,
            input_1=y_true,
            input_2=y_pred,
        )
        sensitivity_by_group["ci"] = ci

    return pd.Series(sensitivity_by_group)


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
