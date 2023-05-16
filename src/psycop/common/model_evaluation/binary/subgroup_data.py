from collections.abc import Sequence
from typing import Optional

import pandas as pd
from psycop.common.model_evaluation.binary.utils import (
    auroc_by_group,
    auroc_within_group,
)
from psycop.common.model_evaluation.utils import bin_continuous_data
from psycop.common.model_training.training_output.dataclasses import EvalDataset


def get_auroc_by_input_df(
    eval_dataset: EvalDataset,
    input_values: pd.Series,
    input_name: str,
    bins: Sequence[float] = (0, 1, 2, 5, 10),
    bin_continuous_input: Optional[bool] = True,
    confidence_interval: bool = True,
) -> pd.DataFrame:
    """Calculate performance by given input values, e.g. age or number of hbac1
    measurements.
    Args:
        eval_dataset: EvalDataset object
        input_values: Input values to calculate performance by
        input_name: Name of the input. Used for column name in output.
        bins: Bins to group by. Defaults to (0, 1, 2, 5, 10, 100).
        bin_continuous_input: Whether to bin input. Defaults to True.
        confidence_interval: Whether to bootstrap confidence interval. Defaults to True.
    Returns:
        pd.DataFrame: Dataframe ready for plotting
    """
    df = pd.DataFrame(
        {
            "y": eval_dataset.y,
            "y_hat": eval_dataset.y_hat_probs,
            input_name: input_values,
        },
    )
    if bin_continuous_input:
        df[f"{input_name}_binned"], _ = bin_continuous_data(df[input_name], bins=bins)

    output_df = auroc_by_group(
        df=df,
        groupby_col_name=f"{input_name}_binned",
        confidence_interval=confidence_interval,
    )

    final_df = output_df.reset_index().rename({0: "metric"}, axis=1)
    return final_df
