import numpy as np
import pandas as pd
from psycop.common.model_evaluation.binary.bootstrap_estimates import (
    bootstrap_estimates,
)
from sklearn.metrics import recall_score, roc_auc_score


def auroc_within_group(
    df: pd.DataFrame,
    confidence_interval: bool = True,
    n_bootstraps: int = 100,
) -> pd.Series:
    """Get the auroc within a dataframe."""
    if df.empty or df["y"].nunique() == 1 or len(df) < 5:
        # Many of our models probably try to predict the majority class.
        # I'm not sure how exactly we want to handle this, but thousands of msg.info is not ideal.
        # For now, suppressing this message.
        # Also protect against fewer than 5 in bin
        return pd.Series({"auroc": np.nan, "n_in_bin": np.nan})

    auroc = roc_auc_score(df["y"], df["y_hat_score"])
    auroc_by_group = {"auroc": auroc, "n_in_bin": len(df)}

    if confidence_interval:
        ci = bootstrap_estimates(
            roc_auc_score,
            n_bootstraps=n_bootstraps,
            ci_width=0.95,
            input_1=df["y"],
            input_2=df["y_hat_score"],
        )
        auroc_by_group["ci_lower"] = ci[0][0]
        auroc_by_group["ci_upper"] = ci[0][1]

    return pd.Series(auroc_by_group)


def auroc_by_group(
    df: pd.DataFrame,
    groupby_col_name: str,
    confidence_interval: bool = True,
    n_bootstraps: int = 100,
) -> pd.DataFrame:
    """Get the auroc by group within a dataframe."""
    return df.groupby(groupby_col_name).apply(
        auroc_within_group,
        confidence_interval=confidence_interval,
        n_bootstraps=n_bootstraps,
    )


def sensitivity_within_group(
    df: pd.DataFrame,
    confidence_interval: bool = True,
    n_bootstraps: int = 100,
) -> pd.Series:
    """Get the sensitivity within a dataframe."""
    if df.empty or len(df) < 5:
        # Protect against fewer than 5 in bin
        return pd.Series({"sensitivity": np.nan, "n_in_bin": np.nan})

    sensitivity = recall_score(df["y"], df["y_hat"])
    sensitivity_by_group = {"sensitivity": sensitivity, "n_in_bin": len(df)}

    if confidence_interval:
        ci = bootstrap_estimates(
            recall_score,
            n_bootstraps=n_bootstraps,
            ci_width=0.95,
            input_1=df["y"],
            input_2=df["y_hat"],
        )
        sensitivity_by_group["ci_lower"] = ci[0][0]
        sensitivity_by_group["ci_upper"] = ci[0][1]

    return pd.Series(sensitivity_by_group)


def sensitivity_by_group(
    df: pd.DataFrame,
    groupby_col_name: str,
    confidence_interval: bool = True,
    n_bootstraps: int = 100,
) -> pd.DataFrame:
    """Get the sensitivity by group within a dataframe."""
    return df.groupby(groupby_col_name).apply(
        sensitivity_within_group,
        confidence_interval=confidence_interval,
        n_bootstraps=n_bootstraps,
    )


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
