import datetime as dt
from typing import List, Tuple, Union

import pandas as pd
from src.utils import convert_all_to_binary, drop_records_if_datediff_days_smaller_than
from wasabi import msg


def process_combined(
    df_combined: pd.DataFrame,
    outcome_col_name: str,
    min_lookahead_days: Union[int, None] = None,
    min_lookbehind_days: Union[int, None] = None,
    cols_to_drop_before_training: List[str] = None,
    convert_all_cols_to_binary: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Convert timestamps to dt., and then to ordinal
    timestamp_colnames = [col for col in df_combined.columns if "timestamp" in col]

    for colname in timestamp_colnames:
        if df_combined[colname].dtype != "datetime64[ns]":
            # Convert all 0s in colname to NaT
            df_combined[colname] = df_combined[colname].apply(
                lambda x: pd.NaT if x == "0" else x
            )

            df_combined[colname] = pd.to_datetime(df_combined[colname])

    if min_lookahead_days is not None:
        drop_if_not_fulfilling_lookahead_days(df_combined, min_lookahead_days)

    if min_lookbehind_days is not None:
        drop_if_not_fulfilling_lookbehind_days(df_combined, min_lookbehind_days)

    # Drop cols that won't generalise
    msg.info("Dropping columns that won't generalise")

    if convert_all_cols_to_binary:
        convert_all_to_binary(df_combined, skip=["age_in_years", "male"])

    msg.info("Converting timestamps to ordinal")
    for colname in timestamp_colnames:
        df_combined[colname] = df_combined[colname].map(dt.datetime.toordinal)

    df_combined.drop(
        cols_to_drop_before_training=cols_to_drop_before_training,
        axis=1,
        errors="ignore",
        inplace=True,
    )

    _X = df_combined.drop(outcome_col_name, axis=1)
    _y = df_combined[outcome_col_name]

    return _X, _y


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


def train(
    train_combined: pd.DataFrame,
    outcome_col_name: str,
    min_lookahead_days: Union[int, None] = None,
    min_lookbehind_days: Union[int, None] = None,
    cols_to_drop_before_training: List[str] = None,
    convert_all_cols_to_binary: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = process_combined(
        df_combined=train_combined,
        outcome_col_name=outcome_col_name,
        min_lookahead_days=min_lookahead_days,
        min_lookbehind_days=min_lookbehind_days,
        cols_to_drop_before_training=cols_to_drop_before_training,
        convert_all_cols_to_binary=convert_all_cols_to_binary,
    )

    return df


def val(
    val_combined: pd.DataFrame,
    outcome_col_name: str,
    min_lookahead_days: Union[int, None] = None,
    min_lookbehind_days: Union[int, None] = None,
    cols_to_drop_before_training: List[str] = None,
    convert_all_cols_to_binary: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = process_combined(
        df_combined=val_combined,
        outcome_col_name=outcome_col_name,
        min_lookahead_days=min_lookahead_days,
        min_lookbehind_days=min_lookbehind_days,
        cols_to_drop_before_training=cols_to_drop_before_training,
        convert_all_cols_to_binary=convert_all_cols_to_binary,
    )

    return df
