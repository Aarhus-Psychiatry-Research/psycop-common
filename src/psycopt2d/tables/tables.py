from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv(Path("tests") / "test_data" / "synth_data.csv")


def auc_by_group_table(
    df: pd.DataFrame,
    pred_probs_col_name: str,
    outcome_col_name: str,
    categorical_groups: Union[List[str], str],
    age_col_name: Optional[str] = None,
    age_bins: Optional[List[int]] = [0, 18, 30, 50, 70, 120],
):

    # Create age bin
    if age_col_name:
        age = bin_age(df[age_col_name], bins=age_bins)
        categorical_groups.append("Age group")
        df["Age group"] = age

    # Group by the groups/bins
    summarize_performance_fn = partial(
        _calc_auc_and_n,
        pred_probs_col_name=pred_probs_col_name,
        outcome_col_name=outcome_col_name,
    )

    groups_df = []
    for group in categorical_groups:
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
) -> pd.DataFrame:
    auc = roc_auc_score(df[outcome_col_name], df[pred_probs_col_name])
    n = len(df)
    return pd.Series([auc, n], index=["AUC", "N"])


def bin_age(series: pd.Series, bins: List[int]) -> pd.Series:
    """For prettier formatting of continuous bins.

    Args:
        series (pd.Series): Series with continuous data such as age
        bins (List[int]): Desired bins

    Returns:
        pd.Series: Binned data

    Example:
    >>> ages = pd.Series([15, 18, 20, 30, 32, 40, 50, 60, 61])
    >>> age_bins = [0, 18, 30, 50, 110]
    >>> bin_Age(ages, age_bins)
    0     0-18
    1     0-18
    2    19-30
    3    19-30
    4    31-50
    5    31-50
    6    31-50
    7      51+
    8      51+
    """
    labels = []
    for i, bin in enumerate(bins):
        if i == 0:
            labels.append(f"{bin}-{bins[i+1]}")
        elif i < len(bins) - 2:
            labels.append(f"{bin+1}-{bins[i+1]}")
        elif i == len(bins) - 2:
            labels.append(f"{bin+1}+")
        else:
            continue

    return pd.cut(series, bins=bins, labels=labels)
