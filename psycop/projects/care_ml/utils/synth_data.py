"""Generate synthetic data for evaluation of the model."""
import datetime as dt
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from wasabi import Printer


def add_age_is_female(
    df: pd.DataFrame,
    id_column_name: str = "dw_ek_borger",
) -> pd.DataFrame:
    """Add age and gender columns to dataframe.

    Args:
        df (pd.DataFrame): The dataframe to add age
        id_column_name (str, optional): The column name of the id column. Defaults to "dw_ek_borger".
    """
    ids = pd.DataFrame({id_column_name: df[id_column_name].unique()})
    ids["age"] = np.random.randint(18, 95, len(ids))
    ids["is_female"] = [np.random.randint(0, 2) for _ in range(0, len(ids))]

    return df.merge(ids)


def days_to_seconds(days: float) -> float:
    """Calculates number of seconds in a number of days.

    Args:
        days (int): Number of days.

    Returns:
        float: Number of seconds
    """
    return days * 24 * 60 * 60


def return_0_with_prob(prob: float) -> Literal[0, 1]:
    """Return nan with a given probability."""

    return 0 if np.random.random() < prob else 1


def null_series_with_prob(
    series: pd.Series,
    prob: float,
    null_value: Any = np.NaN,
) -> Union[pd.Series, None]:
    """Overwrite values in series with null_value with a given probability.

    Args:
        series (pd.Series): Series.
        prob (float): The probability of overwriting all with null_value.
        null_value (any, optional): The value to overwrite with. Defaults to np.NaN.

    Returns:
        A pd.Series.
    """

    if return_0_with_prob(prob) == 0:
        # Replace all values in series with null_value
        series.loc[:] = null_value  # type: ignore
        return series

    return series


def overwrite_prop_with_null(
    series: pd.Series,
    prop: float,
    null_value: Optional[Any] = np.NaN,
) -> pd.Series:
    """Overwrite a proportion of all values in a series with a null value (NaN
    or NaT).

    Args:
        series (pd.Series): The series to overwrite in.
        prop (float): How large a proportion to overwrite.
        null_value (any, optional): The value to overwrite with. Defaults to np.NaN.

    Returns:
        A pd.Series.
    """
    series.loc[
        np.random.choice(series.index, int(len(series) * prop), replace=False)
    ] = null_value

    return series


def random_timestamp(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    number_of_timestamps: int,
) -> pd.DatetimeIndex:
    """Generate a range of random timestamps within a start and end time.

    Args:
        start_time (pd.Timestamp): Earlist possible timestamp
        end_time (pd.Timestamp): Latest possible timestamp
        number_of_timestamps (int): Number of timestamps to generate

    Returns:
        pd.DatetimeIndex: Index of random timestamps
    """
    np.random.seed = 42
    start = start_time.value
    end = end_time.value

    return pd.to_datetime(
        np.random.randint(start, end, number_of_timestamps, dtype=np.int64),
    )


def synth_pred_times(df: pd.DataFrame, pred_hour: int = 6) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        pred_hour (Optional[int], optional): _description_. Defaults to 6.

    Returns:
        pd.DataFrame: _description_
    """
    df["timestamp"] = [
        pd.date_range(
            row["admission_timestamp"].date(),
            row["admission_timestamp"].date()
            + pd.Timedelta(days=int(np.random.normal(16, 5))),
        )
        for idx, row in df.iterrows()
    ]
    df = df.explode("timestamp")

    df = df.assign(timestamp=lambda x: x["timestamp"] + pd.Timedelta(hours=pred_hour))

    return df


if __name__ == "__main__":
    msg = Printer(timestamp=True)
    N_ROWS = 1000

    df = pd.DataFrame()

    df["dw_ek_borger"] = [np.random.randint(0, N_ROWS) for _ in range(N_ROWS)]

    msg.info("Generating synth admission times")
    df["admission_timestamp"] = random_timestamp(
        pd.Timestamp("2015-01-01"),
        pd.Timestamp("2020-01-01"),
        N_ROWS,
    )

    msg.info("Generating synth outcome timestamps")
    df["outcome_timestamp"] = df.groupby("dw_ek_borger")[  # type: ignore
        "admission_timestamp"
    ].transform("min") + dt.timedelta(
        seconds=np.random.randint(0, days_to_seconds(days=5)),  # type: ignore
    )

    df["outcome_timestamp"] = (
        df.groupby("dw_ek_borger")["outcome_timestamp"]
        .apply(
            lambda x: null_series_with_prob(x, prob=0.85),
        )
        .reset_index(drop=True)
    )

    # True label
    df["label"] = df["outcome_timestamp"].notnull().astype(int)

    msg.info("Generating synth prediction times")
    df = synth_pred_times(df)

    msg.info("Generating synth model predictions")
    df["pred_prob"] = [(np.random.random() - 0.45) for _ in range(len(df))]
    df["pred_prob"] = df["pred_prob"].clip(0, 1)
    df["pred"] = df["pred_prob"].clip(0, 1).round()

    df = add_age_is_female(df)

    df.to_csv(Path("tests") / "test_data" / "model_eval" / "synth_eval_data.csv")
