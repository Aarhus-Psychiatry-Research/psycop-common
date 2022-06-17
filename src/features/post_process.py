import datetime as dt
from typing import List, Tuple, Union

import pandas as pd
from src.utils import convert_all_to_binary, drop_records_if_datediff_days_smaller_than
from wasabi import msg
import datetime as dt

from psycopmlutils.loaders import sql_load


def combined(
    X: pd.DataFrame,
    y: pd.Series,
    outcome_col_name: str,
    min_lookahead_days: Union[int, None] = None,
    min_lookbehind_days: Union[int, None] = None,
    convert_all_cols_to_binary: bool = False,
    cols_to_drop: List[str] = None,
    drop_if_any_diabetes_before_date: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df_combined = X
    df_combined[outcome_col_name] = y

    if drop_if_any_diabetes_before_date is not None:
        df_combined = drop_ids_if_any_outcome_before_date(
            date_str=drop_if_any_diabetes_before_date, df=df_combined
        )

    if min_lookahead_days is not None:
        drop_if_not_fulfilling_lookahead_days(df_combined, min_lookahead_days)

    if min_lookbehind_days is not None:
        drop_if_not_fulfilling_lookbehind_days(df_combined, min_lookbehind_days)

    # Dataframe for eval, make copies before making transformations
    # that make interpretation more difficult
    _X_eval = df_combined.copy().drop(outcome_col_name, axis=1)
    _y_eval = df_combined.copy()[outcome_col_name]

    ## Transformations that make eval harder
    # Convert timestamps to ordinal
    timestamp_colnames = [col for col in df_combined.columns if "timestamp" in col]
    msg.info("Converting timestamps to ordinal")
    for colname in timestamp_colnames:
        df_combined[colname] = df_combined[colname].map(dt.datetime.toordinal)

    if cols_to_drop is not None:
        df_combined.drop(columns=cols_to_drop, inplace=True, axis=1)

    if convert_all_cols_to_binary:
        convert_all_to_binary(
            df_combined, skip=["age_in_years", "male", outcome_col_name]
        )

    _X = df_combined.drop(outcome_col_name, axis=1)
    _y = df_combined[outcome_col_name]

    return _X, _y, _X_eval, _y_eval


def drop_ids_if_any_outcome_before_date(df, date_str):
    # Convert drop_if_any_diabetes_before_date string to datetime
    date_str = dt.datetime.strptime(date_str, "%Y-%m-%d")

    df_any_diab = sql_load(
        query=f"SELECT dw_ek_borger, datotid_first_diabetes_any FROM [fct].[psycop_t2d_first_diabetes_any]",
        format_timestamp_cols_to_datetime=True,
    )

    df_any_diab["date_str"] = date_str

    # Keep all rows in df_timestamp_any_diabetes where "datotid_diabetes_any" is smaller than dt_date_str
    df_diab_before_date_str = df_any_diab[
        df_any_diab["datotid_first_diabetes_any"] < df_any_diab["date_str"]
    ]

    # Get the union of the two dataframes, add an indicator.
    # If the dw_ek_borger is only in df_combined, that's the one to keep
    df = df.merge(
        df_diab_before_date_str,
        on="dw_ek_borger",
        how="outer",
        indicator="drop_because_any_diabetes_before_date",
    )

    # Keep only rows from df_combined where drop_because_any_diabetes_before_date == "left_only"
    df = df[df["drop_because_any_diabetes_before_date"] == "left_only"]

    df.drop(
        columns=["date_str", "drop_because_any_diabetes_before_date"],
        axis=1,
        inplace=True,
    )

    return df


def drop_if_not_fulfilling_lookbehind_days(df_combined, min_lookbehind_days):
    _first_prediction_time_col_name = "first_prediction_time"
    df_combined[_first_prediction_time_col_name] = df_combined["timestamp"].min()

    drop_records_if_datediff_days_smaller_than(
        df=df_combined,
        t2_col_name="timestamp",
        t1_col_name=_first_prediction_time_col_name,
        threshold_days=min_lookbehind_days,
        inplace=True,
    )

    df_combined.drop(_first_prediction_time_col_name, inplace=True, axis=1)


def drop_if_not_fulfilling_lookahead_days(df_combined, min_lookahead_days):
    _last_prediction_time_col_name = "last_prediction_time"
    df_combined[_last_prediction_time_col_name] = df_combined["timestamp"].max()

    drop_records_if_datediff_days_smaller_than(
        df=df_combined,
        t2_col_name=_last_prediction_time_col_name,
        t1_col_name="timestamp",
        threshold_days=min_lookahead_days,
        inplace=True,
    )

    df_combined.drop(_last_prediction_time_col_name, inplace=True, axis=1)
