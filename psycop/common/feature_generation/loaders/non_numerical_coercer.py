"""Some of our data series contain both numerical values and strings like
'>=12'.

This function takes a df and coerces all strings into numerical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def multiply_inequalities_in_df(
    df: pd.DataFrame,
    round_to_decimals: int = 6,
    ineq2mult: dict[str, float] | None = None,
    col_to_multiply: str = "value",
) -> pd.DataFrame:
    """Multiply inequalities in a dataframe by a factor.

    Args:
        df (pd.Dataframe): The dataframe to be modified.
        round_to_decimals (int): How many decimals to round the value to.
        ineq2mult (dict[str, float]): A dictionary with the inequalities as keys and the factors as values.
            Current values are arbitrary, but ensure that inequalities are somewhat separated from the continuous part of the distribution.
        col_to_multiply (str): The column to multiply.

    Returns:
        pd.Dataframe: The modified dataframe.
    """
    if ineq2mult is None:  # Avoid sharing a mutable keyword argument between function invocations
        ineq2mult = {"<": 0.67, "<=": 0.8, ">": 1.5, ">=": 1.2}

    # Sort inequalities by length, so that we don't replace "<" in "<=".
    in_eqs = sorted(ineq2mult.keys(), key=len, reverse=True)

    for in_eq in in_eqs:
        col_can_contain_ineq = df[col_to_multiply].dtype == "object"

        if col_can_contain_ineq:
            starts_with_ineq_idxs = df[col_to_multiply].str.startswith(in_eq).fillna(False)

            df.loc[starts_with_ineq_idxs, col_to_multiply] = (
                df.loc[starts_with_ineq_idxs, col_to_multiply]
                .str.replace(",", ".")
                .str.extract(r"(\d+\.\d+|\d+)", expand=False)
                .astype(float)
                .mul(ineq2mult[in_eq])
                .round(round_to_decimals)
            )

    # Convert col_to_multiply dtype to float
    df[col_to_multiply] = df[col_to_multiply].astype(float)

    return df
