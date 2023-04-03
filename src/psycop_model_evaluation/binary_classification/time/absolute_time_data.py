from collections.abc import Iterable

import pandas as pd
from psycop_model_evaluation.binary_classification.utils import (
    calc_performance,
)
from sklearn.metrics import roc_auc_score


def create_roc_auc_by_absolute_time_df(
    labels: Iterable[int],
    y_hat: Iterable[float],
    timestamps: Iterable[pd.Timestamp],
    bin_period: str,
) -> pd.DataFrame:
    """Calculate performance by calendar time of prediction.
    Args:
        labels (Iterable[int]): True labels
        y_hat (Iterable[int, float]): Predicted probabilities or labels depending on metric
        timestamps (Iterable[pd.Timestamp]): Timestamps of predictions
        bin_period (str): How to bin time. Takes "M" for month, "Q" for quarter or "Y" for year
    Returns:
        pd.DataFrame: Dataframe ready for plotting
    """
    df = pd.DataFrame({"y": labels, "y_hat": y_hat, "timestamp": timestamps})

    df["time_bin"] = pd.PeriodIndex(df["timestamp"], freq=bin_period).format()

    output_df = df.groupby("time_bin").apply(
        func=calc_performance,  # type: ignore
        metric=roc_auc_score,
    )

    return output_df.reset_index().rename({0: "metric"}, axis=1)
