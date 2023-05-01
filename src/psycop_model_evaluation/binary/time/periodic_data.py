from collections.abc import Iterable

import pandas as pd
from psycop_model_evaluation.binary.utils import (
    calc_performance,
)
from sklearn.metrics import roc_auc_score


def roc_auc_by_periodic_time_df(
    labels: Iterable[int],
    y_hat: Iterable[float],
    timestamps: Iterable[pd.Timestamp],
    bin_period: str,
) -> pd.DataFrame:
    """Calculate performance by cyclic time period of prediction time data
    frame. Cyclic time periods include e.g. day of week, hour of day, etc.
    Args:
        labels (Iterable[int]): True labels
        y_hat (Iterable[int, float]): Predicted probabilities or labels depending on metric
        timestamps (Iterable[pd.Timestamp]): Timestamps of predictions
        bin_period (str): Which cyclic time period to bin on. Takes "H" for hour of day, "D" for day of week and "M" for month of year.
    Returns:
        pd.DataFrame: Dataframe ready for plotting
    """
    df = pd.DataFrame({"y": labels, "y_hat": y_hat, "timestamp": timestamps})

    if bin_period == "H":
        df["time_bin"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H")
    elif bin_period == "D":
        df["time_bin"] = pd.to_datetime(df["timestamp"]).dt.strftime("%A")
        # Sort days of week correctly
        df["time_bin"] = pd.Categorical(
            df["time_bin"],
            categories=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            ordered=True,
        )
    elif bin_period == "M":
        df["time_bin"] = pd.to_datetime(df["timestamp"]).dt.strftime("%B")
        # Sort months correctly
        df["time_bin"] = pd.Categorical(
            df["time_bin"],
            categories=[
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ],
            ordered=True,
        )
    else:
        raise ValueError(
            "bin_period must be 'H' for hour of day, 'D' for day of week or 'M' for month of year",
        )

    output_df = df.groupby("time_bin").apply(
        func=calc_performance,  # type: ignore
        metric=roc_auc_score,
    )

    return output_df.reset_index().rename({0: "metric"}, axis=1)
