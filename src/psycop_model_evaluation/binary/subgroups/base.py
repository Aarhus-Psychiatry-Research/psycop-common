from collections.abc import Sequence
from functools import partial
from typing import Optional

import pandas as pd
from psycop_model_evaluation.binary.utils import (
    calc_performance,
)
from psycop_model_evaluation.utils import bin_continuous_data
from psycop_model_training.training_output.dataclasses import EvalDataset
from sklearn.metrics import roc_auc_score


def create_roc_auc_by_input(
    eval_dataset: EvalDataset,
    input_values: Sequence[float],
    input_name: str,
    bins: Sequence[float] = (0, 1, 2, 5, 10),
    bin_continuous_input: Optional[bool] = True,
    confidence_interval: Optional[float] = None,
) -> pd.DataFrame:
    """Calculate performance by given input values, e.g. age or number of hbac1
    measurements.
    Args:
        eval_dataset: EvalDataset object
        input_values: Input values to calculate performance by
        input_name: Name of the input
        bins: Bins to group by. Defaults to (0, 1, 2, 5, 10, 100).
        bin_continuous_input: Whether to bin input. Defaults to True.
        confidence_interval: Confidence interval for calculating. Defaults to None.
            in which case the no confidence interval is calculated.
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
    calc_perf = partial(
        calc_performance,
        metric=roc_auc_score,
        confidence_interval=confidence_interval,
    )
    if bin_continuous_input:
        df[f"{input_name}_binned"], _ = bin_continuous_data(df[input_name], bins=bins)

        output_df = df.groupby(f"{input_name}_binned").apply(
            calc_perf,  # type: ignore
        )

    else:
        output_df = df.groupby(input_name).apply(calc_perf)  # type: ignore

    final_df = output_df.reset_index().rename({0: "metric"}, axis=1)
    return final_df
