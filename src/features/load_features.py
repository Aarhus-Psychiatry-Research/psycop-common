import datetime as dt
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from psycopmlutils.loaders import sql_load
from wasabi import Printer

msg = Printer(timestamp=True)


def load_datasets(
    dataset_name: str,
    outcome_col_name: str,
    n_to_load: Union[None, int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    sql_table_name = f"psycop_t2d_{dataset_name}"

    if n_to_load is not None:
        msg.info(f"{sql_table_name}: Loading {n_to_load} rows from")
        select = f"SELECT TOP {n_to_load}"
    else:
        msg.info(f"{sql_table_name}: Loading all rows")
        select = "SELECT"

    df_combined = sql_load(
        query=f"{select} * FROM [fct].[{sql_table_name}]",
        format_timestamp_cols_to_datetime=False,
    )

    msg.info(f"{sql_table_name}: Finished loading row_")

    # Convert timestamps
    timestamp_colnames = [col for col in df_combined.columns if "timestamp" in col]

    for colname in timestamp_colnames:
        if df_combined[colname].dtype != "datetime64[ns]":
            # Convert all 0s in colname to NaT
            df_combined[colname] = df_combined[colname].apply(
                lambda x: pd.NaT if x == "0" else x
            )

            df_combined[colname] = pd.to_datetime(df_combined[colname])

    _X = df_combined.drop(outcome_col_name, axis=1)
    _y = df_combined[outcome_col_name]

    msg.good(f"{dataset_name}: Returning!")
    return _X, _y


def load_train(outcome_col_name, n_to_load: Union[None, int] = None):
    return load_datasets(
        "train", outcome_col_name=outcome_col_name, n_to_load=n_to_load
    )


def load_test(outcome_col_name, n_to_load: Union[None, int] = None):
    return load_datasets("test", outcome_col_name=outcome_col_name, n_to_load=n_to_load)


def load_val(outcome_col_name, n_to_load: Union[None, int] = None):
    return load_datasets("val", outcome_col_name=outcome_col_name, n_to_load=n_to_load)
