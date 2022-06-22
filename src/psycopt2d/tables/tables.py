from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

df = pd.read_csv(Path("tests") / "test_data" / "synth_data.csv")


def add_age_gender(df):
    ids = pd.DataFrame({"dw_ek_borger": df["dw_ek_borger"].unique()})
    ids["age"] = np.random.randint(17, 95, len(ids))
    ids["gender"] = np.where(ids["dw_ek_borger"] > 30_000, "F", "M")

    return df.merge(ids)


def grouped_performance_table(
    df: pd.DataFrame,
    predictions_col_name: str,
    prediction_probabilities_col_name: str,
    outcome_col_name: str,
    groups: Union[List[str], str],
    age_col_name: Optional[str] = None,
    age_bins: Optional[List[int]] = None,
):

    # Create age bin
    if age_col_name:
        age = bin_age(df[age_col_name], bins=age_bins)
        groups = groups.append("age")
        df["age"] = age

    # Group by the groups/bins
    summarize_performance_fn = partial(
        summarize_performance,
        predictions_col_name=predictions_col_name,
        prediction_probabilities_col_name=prediction_probabilities_col_name,
        outcome_col_name=outcome_col_name,
    )

    df.groupby(groups).apply(summarize_performance_fn)

    # Calculate auc, sens, spec, ppv, npv

    # Rows groups, columns are metrics
    pass


def summarize_performance(
    df: pd.DataFrame,
    predictions_col_name: str,
    prediction_probabilities_col_name: str,
    outcome_col_name: str,
) -> pd.DataFrame:
    pass


def bin_age(series: pd.Series, bins: List[int]) -> pd.Series:
    """Maybe use for prettier formatting. Otherwise, just use default.

    Args:
        series (pd.Series): Series with continuous data such as age
        bins (List[int]): Desired bins

    Returns:
        pd.Series: Binned data
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


if __name__ == "__main__":

    df = pd.read_csv(Path("tests") / "test_data" / "synth_data.csv")
    df = add_age_gender(df)

    age_bins = [0, 18, 30, 50, 70, 100]
    age_labels = ["0-18", "19-30", "31-50", "51-70", "71-99"]
    df["age_bins__"] = bin_age(df["age"], age_bins)

    df["age_bins"] = pd.cut(df["age"], bins=age_bins)
