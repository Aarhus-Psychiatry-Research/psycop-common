"""Tables for description and evaluation of models and patient population."""
from typing import Union

import pandas as pd
import wandb


def output_table(
    output_format: str,
    df: pd.DataFrame,
) -> Union[pd.DataFrame, wandb.Table, str]:
    """Output table in specified format."""
    if output_format == "html":
        return df.reset_index(drop=True).to_html()
    if output_format == "df":
        return df.reset_index(drop=True)
    if output_format == "wandb_table":
        return wandb.Table(dataframe=df)

    raise ValueError("Output format does not match anything that is allowed")
