from typing import List, Tuple, Union
import pandas as pd
from src.utils import (
    drop_records_if_datediff_days_smaller_than,
)

from wasabi import msg


def process_combined(
    df_combined: pd.DataFrame,
    min_lookahead_days: Union[int, None] = None,
    min_lookbehind_days: Union[int, None] = None,
    cols_to_drop_before_training: List[str] = None,
):
    # Convert timestamps
    timestamp_colnames = [col for col in df_combined.columns if "timestamp" in col]

    for colname in timestamp_colnames:
        if df_combined[colname].dtype != "datetime64[ns]":
            # Convert all 0s in colname to NaT
            df_combined[colname] = df_combined[colname].apply(
                lambda x: pd.NaT if x == "0" else x
            )

            df_combined[colname] = pd.to_datetime(df_combined[colname])

    if min_lookahead_days is not None:
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

    if min_lookbehind_days is not None:
        _first_prediction_time_col_name = "first_prediction_time"
        df_combined[_first_prediction_time_col_name] = df_combined["timestamp"].min()

        drop_records_if_datediff_days_smaller_than(
            df=df_combined,
            t2_col_name="timestamp",
            t1_col_name=_first_prediction_time_col_name,
            threshold_days=cfg.preprocessing.min_lookbehind_days,
            inplace=True,
        )

        df_combined.drop(_first_prediction_time_col_name, inplace=True, axis=1)

    # Drop cols that won't generalise
    msg.info("Dropping columns that won't generalise")

    cols_to_drop_before_training = [
        "dw_ek_borger",
        "prediction_time_uuid",
        outcome_timestamp_col_name,
    ]

    ds.drop(cols_to_drop_before_training, axis=1, errors="ignore", inplace=True)


def process_train(
    df_train_combined: pd.DataFrame, min_lookahead_days
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return process_combined(df_train_combined)


def process_val(df_val_combined: pd.DataFrame):
    raise NotImplementedError
