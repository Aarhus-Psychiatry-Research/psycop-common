from collections.abc import Iterable

import pandas as pd
from psycop.common.model_evaluation.binary.utils import auroc_by_group


def create_roc_auc_by_absolute_time_df(
    labels: Iterable[int],
    y_hat: Iterable[float],
    timestamps: Iterable[pd.Timestamp],
    bin_period: str,
    confidence_interval: bool = True,
    n_bootstraps: int = 100,
) -> pd.DataFrame:
    """Calculate performance by calendar time of prediction.
    Args:
        labels: True labels
        y_hat: Predicted probabilities or labels depending on metric
        timestamps: Timestamps of predictions
        bin_period: How to bin time. Takes "M" for month, "Q" for quarter or "Y" for year
        confidence_interval: Whether to create bootstrapped confidence interval.
        n_bootstraps: Number of bootstraps to use for confidence interval.
    Returns:
        Dataframe ready for plotting
    """
    df = pd.DataFrame({"y": labels, "y_hat": y_hat, "timestamp": timestamps})

    df["time_bin"] = pd.PeriodIndex(df["timestamp"], freq=bin_period).format()

    output_df = df.groupby("time_bin").apply(
        func=auroc_by_group,  # type: ignore
        y_true=df["y"],
        y_pred_proba=df["y_hat"],
        confidence_interval=confidence_interval,
        n_bootstraps=n_bootstraps,
    )

    return output_df.reset_index().rename({0: "metric"}, axis=1)
