from collections.abc import Iterable
from functools import partial
from typing import Optional

import pandas as pd
from psycop_model_evaluation.binary.utils import (
    calc_performance,
)
from sklearn.metrics import roc_auc_score


def create_roc_auc_by_absolute_time_df(
    labels: Iterable[int],
    y_hat: Iterable[float],
    timestamps: Iterable[pd.Timestamp],
    bin_period: str,
    confidence_interval: Optional[float] = None,
) -> pd.DataFrame:
    """Calculate performance by calendar time of prediction.
    Args:
        labels: True labels
        y_hat: Predicted probabilities or labels depending on metric
        timestamps: Timestamps of predictions
        bin_period: How to bin time. Takes "M" for month, "Q" for quarter or "Y" for year
        confidence_interval: Confidence interval to use for create the bootstrapped confidence intervals
    Returns:
        Dataframe ready for plotting
    """
    df = pd.DataFrame({"y": labels, "y_hat": y_hat, "timestamp": timestamps})

    df["time_bin"] = pd.PeriodIndex(df["timestamp"], freq=bin_period).format()

    _calc_performance = partial(
        calc_performance,
        metric=roc_auc_score,
        confidence_interval=confidence_interval,
    )

    output_df = df.groupby("time_bin").apply(_calc_performance)

    return output_df.reset_index().rename({0: "metric"}, axis=1)
