"""Generate synthetic data for evaluation of the model."""
import datetime as dt
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from wasabi import Printer


def add_age_is_female(df: pd.DataFrame) -> pd.DataFrame:
    """Add age and gender columns to dataframe.

    Args:
        df (pd.DataFrame): The dataframe to add age
    """
    ids = pd.DataFrame({"dw_ek_borger": df["dw_ek_borger"].unique()})
    ids["age"] = np.random.randint(18, 95, len(ids))
    ids["is_female"] = np.where(ids["dw_ek_borger"] > 30_000, 1, 0)

    return df.merge(ids)


def years_to_seconds(years: float) -> float:
    """Calculates number of seconds in a number of years.

    Args:
        years (int): Number of years.

    Returns:
        _type_: _description_
    """
    return years * 365 * 24 * 60 * 60


def return_0_with_prob(prob: float) -> Literal[0, 1]:
    """Return 0 with a given probability."""
    return 0 if np.random.random() < prob else 1


def null_series_with_prob(
    series: pd.Series,
    prob: float,
    null_value: Any = np.NaN,
) -> Union[pd.Series, None]:
    """Overwrite all values in series with null_value with a given probability.

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
    return None


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


if __name__ == "__main__":
    msg = Printer(timestamp=True)
    base_timestamp = pd.Timestamp.today()
    N_ROWS = 100_000

    df = pd.DataFrame()

    df["dw_ek_borger"] = [np.random.randint(0, 100_000) for _ in range(N_ROWS)]

    # Generate timestamps
    df["timestamp"] = [base_timestamp] * N_ROWS

    msg.info("Adding differences")
    df["time_differences"] = [
        dt.timedelta(
            seconds=np.random.randint(  # type: ignore
                years_to_seconds(years=5),  # type: ignore
                years_to_seconds(years=10),  # type: ignore
            ),
        )
        for _ in range(N_ROWS)
    ]
    df["timestamp"] = df["timestamp"] + df["time_differences"]
    df = df.drop("time_differences", axis=1)

    df["pred_prob"] = [(np.random.random() - 0.45) for _ in range(N_ROWS)]
    df["pred_prob"] = df["pred_prob"].clip(0, 1)
    df["pred"] = df["pred_prob"].clip(0, 1).round()

    df["timestamp_first_pred_time"] = df.groupby("dw_ek_borger")["timestamp"].transform(
        "min",
    )

    # Generate t2d timestamps
    msg.info("Generating T2D-timestamps")
    df["timestamp_t2d_diag"] = df.groupby("dw_ek_borger")[  # type: ignore
        "timestamp_first_pred_time"
    ].transform("min") + dt.timedelta(
        seconds=np.random.randint(0, years_to_seconds(years=5)),  # type: ignore
    )
    df["timestamp_t2d_diag"] = df.groupby("dw_ek_borger")["timestamp_t2d_diag"].apply(
        lambda x: null_series_with_prob(x, prob=0.85),
    )

    # Generate first HbA1c timestamps
    msg.info("Generating HbA1c timestamps")
    df["timestamp_first_hba1c"] = df.groupby("dw_ek_borger")[  # type: ignore
        "timestamp_first_pred_time"
    ].transform("min") + dt.timedelta(
        seconds=np.random.randint(0, years_to_seconds(years=4)),  # type: ignore
    )
    df["timestamp_hba1c_copy"] = df["timestamp_first_hba1c"]

    # Replace most with null
    msg.info("Replacing with null")
    df["timestamp_first_hba1c"] = df.groupby("dw_ek_borger")[
        "timestamp_first_hba1c"
    ].apply(lambda x: null_series_with_prob(x, prob=0.95))

    # Put back values if there is a T2D date
    msg.info("Putting back if there is T2D date")
    df["timestamp_first_hba1c"] = df.apply(
        lambda x: x["timestamp_hba1c_copy"]
        if not pd.isnull(x["timestamp_t2d_diag"])
        else x["timestamp_first_hba1c"],
        axis=1,
    )
    df = df.drop("timestamp_hba1c_copy", axis=1)

    # True label
    df["label"] = df["timestamp_t2d_diag"].notnull().astype(int)

    # Round off datetimes to whole minutes
    for col in df.columns:
        if "timestamp" in col:
            df[col] = df[col].dt.round("min")

    df = add_age_is_female(df)

    df.to_csv(Path("tests") / "test_data" / "model_eval" / "synth_eval_data.csv")
