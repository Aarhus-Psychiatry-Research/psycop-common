import logging

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

from psycop.common.model_evaluation.binary.bootstrap_estimates import bootstrap_estimates

log = logging.getLogger(__file__)


def _auroc_within_group(
    df: pd.DataFrame,
    confidence_interval: bool = True,
    n_bootstraps: int = 100,
    stratified: bool = False,
) -> pd.DataFrame:
    """Get the AUROC within a dataframe."""
    if df.empty or df["y"].nunique() == 1 or len(df) < 5:
        # Many of our models probably try to predict the majority class.
        # I'm not sure how exactly we want to handle this, but thousands of msg.info is not ideal.
        # For now, suppressing this message.
        # Also protect against fewer than 5 in bin
        return pd.DataFrame({"auroc": np.nan, "n_in_bin": np.nan}, index=[0])

    auroc = roc_auc_score(df["y"], df["y_hat_probs"])
    auroc_by_group = {"auroc": auroc, "n_in_bin": len(df)}

    if confidence_interval:
        log.info("Bootstrapping estimates")

        ci = bootstrap_estimates(
            roc_auc_score,
            n_bootstraps=n_bootstraps,
            ci_width=0.95,
            input_1=df["y"],
            input_2=df["y_hat_probs"],
            stratified=stratified,
        )
        auroc_by_group["ci_lower"] = max(0.0, ci[0][0])
        auroc_by_group["ci_upper"] = min(1.0, ci[0][1])

    return pd.DataFrame(auroc_by_group, index=[0])


def auroc_by_group(
    df: pd.DataFrame,
    groupby_col_name: str,
    confidence_interval: bool = True,
    n_bootstraps: int = 100,
    stratified: bool = False,
) -> pd.DataFrame:
    """Get the AUROC by group within a dataframe.  If class imbalance is high, the stratified
    argument may be set to true to ensure that each class is represented in each bootstrap sample.
    If not, the scitkit learn statistic functions may silently return NA CIs."""
    df = (
        df.groupby(groupby_col_name).apply(
            _auroc_within_group,
            confidence_interval=confidence_interval,
            n_bootstraps=n_bootstraps,
            stratified=stratified,
        )
    ).reset_index(drop=False)

    return df


def _sensitivity_within_group(
    df: pd.DataFrame,
    confidence_interval: bool = True,
    n_bootstraps: int = 100,
    stratified: bool = False,
) -> pd.DataFrame:
    """Get the sensitivity within a dataframe."""
    if df.empty or len(df) < 5:
        # Protect against fewer than 5 in bin
        return pd.DataFrame({"sensitivity": np.nan, "n_in_bin": np.nan}, index=[0])

    sensitivity = recall_score(df["y"], df["y_hat"])
    sensitivity_by_group = {"sensitivity": sensitivity, "n_in_bin": len(df)}

    if confidence_interval:
        ci = bootstrap_estimates(
            recall_score,
            n_bootstraps=n_bootstraps,
            ci_width=0.95,
            input_1=df["y"],
            input_2=df["y_hat"],
            stratified=stratified,
        )
        sensitivity_by_group["ci_lower"] = max(0.0, ci[0][0])
        sensitivity_by_group["ci_upper"] = min(1.0, ci[0][1])

    return pd.DataFrame(sensitivity_by_group, index=[0])


def sensitivity_by_group(
    df: pd.DataFrame,
    groupby_col_name: str,
    confidence_interval: bool = True,
    n_bootstraps: int = 100,
    stratified: bool = False,
) -> pd.DataFrame:
    """Get the sensitivity by group within a dataframe. If class imbalance is high, the stratified
    argument may be set to true to ensure that each class is represented in each bootstrap sample.
    If not, the scitkit learn statistic functions may silently return NA CIs"""
    df = (
        df.groupby(groupby_col_name)
        .apply(
            func=_sensitivity_within_group,
            confidence_interval=confidence_interval,
            n_bootstraps=n_bootstraps,
            stratified=stratified,
        )
        .reset_index(drop=False)
    )

    return df


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
