from collections.abc import Sequence
from typing import Optional

import pandas as pd

from psycop.common.model_evaluation.binary.utils import _auroc_within_group  # type: ignore
from psycop.common.model_evaluation.utils import bin_continuous_data


def auroc_by_model(
    input_values: pd.Series,  # type: ignore
    y: pd.Series,  # type: ignore
    y_hat_probs: pd.Series,  # type: ignore
    input_name: str,
    bins: Sequence[float] = (0, 1, 2, 5, 10),
    bin_continuous_input: Optional[bool] = True,
    confidence_interval: bool = True,
    n_bootstraps: int = 100,
) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "y_hat_probs": y_hat_probs, input_name: input_values})

    if bin_continuous_input:
        groupby_col_name = f"{input_name}_binned"
        df[groupby_col_name], _ = bin_continuous_data(df[input_name], bins=bins)
    else:
        groupby_col_name = input_name

    output_df = auroc_by_group(
        df=df,
        groupby_col_name=groupby_col_name,
        confidence_interval=confidence_interval,
        n_bootstraps=n_bootstraps,
    )

    final_df = output_df.reset_index().rename({0: "metric"}, axis=1)
    return final_df


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
    rows = []

    for group_value, group_df in df.groupby(groupby_col_name):
        group_result = _auroc_within_group(
            group_df,
            confidence_interval=confidence_interval,
            n_bootstraps=n_bootstraps,
            stratified=stratified,
        ).copy()

        group_result[groupby_col_name] = group_value
        rows.append(group_result)

    result = pd.concat(rows, ignore_index=True)

    # Put grouping column first
    other_cols = [col for col in result.columns if col != groupby_col_name]
    result = result[[groupby_col_name, *other_cols]]

    return result
