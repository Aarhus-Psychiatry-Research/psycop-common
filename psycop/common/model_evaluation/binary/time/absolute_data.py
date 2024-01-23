from collections.abc import Iterable

import pandas as pd

from psycop.common.model_evaluation.binary.utils import auroc_by_group


def create_roc_auc_by_absolute_time_df(
    labels: Iterable[int],
    y_hat_probs: Iterable[float],
    timestamps: Iterable[pd.Timestamp],
    bin_period: str,
    confidence_interval: bool = True,
    n_bootstraps: int = 100,
) -> pd.DataFrame:
    """Calculate performance by calendar time of prediction.
    Args:
        labels: True labels
        y_hat_probs: Predicted probabilities or labels depending on metric
        timestamps: Timestamps of predictions
        bin_period: How to bin time. Takes "M" for month, "Q" for quarter or "Y" for year
        confidence_interval: Whether to create bootstrapped confidence interval.
        n_bootstraps: Number of bootstraps to use for confidence interval.
    Returns:
        Dataframe ready for plotting
    """
    df = pd.DataFrame({"y": labels, "y_hat_probs": y_hat_probs, "timestamp": timestamps})

    df["time_bin"] = pd.PeriodIndex(df["timestamp"], freq=bin_period).format()

    output_df = auroc_by_group(
        df=df,
        groupby_col_name="time_bin",
        confidence_interval=confidence_interval,
        n_bootstraps=n_bootstraps,
    )

    return output_df.reset_index().rename({0: "metric"}, axis=1)
