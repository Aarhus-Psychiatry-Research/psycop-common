from collections.abc import Sequence
from typing import Optional

import pandas as pd
from psycop_model_evaluation.binary.utils import (
    calc_performance,
)
from psycop_model_evaluation.utils import bin_continuous_data
from psycop_model_training.model_eval.dataclasses import EvalDataset
from sklearn.metrics import roc_auc_score


def create_roc_auc_by_input(
    eval_dataset: EvalDataset,
    input_values: Sequence[float],
    input_name: str,
    bins: Sequence[float] = (0, 1, 2, 5, 10),
    bin_continuous_input: Optional[bool] = True,
) -> pd.DataFrame:
    """Calculate performance by given input values, e.g. age or number of hbac1
    measurements.
    Args:
        eval_dataset: EvalDataset object
        input_values (Sequence[float]): Input values to calculate performance by
        input_name (str): Name of the input
        bins (Sequence[float]): Bins to group by. Defaults to (0, 1, 2, 5, 10, 100).
        bin_continuous_input (bool, optional): Whether to bin input. Defaults to True.
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

    # bin data and calculate metric per bin
    if bin_continuous_input:
        df[f"{input_name}_binned"], _ = bin_continuous_data(df[input_name], bins=bins)

        output_df = df.groupby(f"{input_name}_binned").apply(
            calc_performance,  # type: ignore
            roc_auc_score,
        )

    else:
        output_df = df.groupby(input_name).apply(calc_performance, roc_auc_score)  # type: ignore

    final_df = output_df.reset_index().rename({0: "metric"}, axis=1)
    return final_df
