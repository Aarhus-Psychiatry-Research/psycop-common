from collections.abc import Sequence
from typing import Optional

import pandas as pd

from psycop.common.model_evaluation.binary.utils import auroc_by_group
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
