from functools import partial
from typing import Iterable, List, Union

import pandas as pd
import wandb
from sklearn.metrics import roc_auc_score


def auc_by_group_table(
    df: pd.DataFrame,
    pred_probs_col_name: str,
    outcome_col_name: str,
    groups: Union[List[str], str],
) -> pd.DataFrame:
    """Create table with AUC per group.

    Args:
        df (pd.DataFrame): DataFrame containing predicted probabilities, labels,
            and groups to stratify by.
        pred_probs_col_name (str): The column containing the predicted probabilities
        outcome_col_name (str): The column containing the labels
        groups (Union[List[str], str]): The (categorical) groups to
            stratify the table by.

    Returns:
        pd.DataFrame: DataFrame with results
    """
    if isinstance(groups, str):
        groups = [groups]

    # Group by the groups/bins
    summarize_performance_fn = partial(
        _calc_auc_and_n,
        pred_probs_col_name=pred_probs_col_name,
        outcome_col_name=outcome_col_name,
    )

    groups_df = []
    for group in groups:
        table = df.groupby(group).apply(summarize_performance_fn)
        # Rename index for consistent naming
        table = table.rename_axis("Value")
        # Add group as index
        table = pd.concat({group: table}, names=["Group"])
        groups_df.append(table)

    return pd.concat(groups_df)


def _calc_auc_and_n(
    df: pd.DataFrame,
    pred_probs_col_name: str,
    outcome_col_name: str,
) -> pd.Series:
    """Calculate auc and number of data points per group.

    Args:
        df (pd.DataFrame): DataFrame containing predicted probabilities and labels
        pred_probs_col_name (str): The column containing predicted probabilities
        outcome_col_name (str): THe containing the labels

    Returns:
        pd.Series: Series containing AUC and N
    """
    auc = roc_auc_score(df[outcome_col_name], df[pred_probs_col_name])
    n = len(df)
    return pd.Series([auc, n], index=["AUC", "N"])


def generate_feature_importances_table(
    column_names: Iterable[str],
    feature_importances: Iterable[str],
    output_format: str = "wandb_table",
) -> Union[pd.DataFrame, str]:

    df = pd.DataFrame(
        {"predictor": column_names, "feature_importance": feature_importances},
    )
    df = df.sort_values("feature_importance", ascending=False)

    if output_format == "html":
        return df.reset_index(drop=True).to_html()
    elif output_format == "df":
        return df.reset_index(drop=True)
    elif output_format == "wandb_table":
        return wandb.Table(dataframe=df)
    else:
        raise ValueError("Output format does not match anything that is allowed")
