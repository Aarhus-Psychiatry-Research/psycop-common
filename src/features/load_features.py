from typing import Tuple, Union

import pandas as pd
from psycopmlutils.loaders import sql_load
from wasabi import Printer

msg = Printer(timestamp=True)


def load_combined(
    split_name: str,
    n_to_load: Union[None, int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    sql_table_name = f"psycop_t2d_{split_name}"

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

    msg.good(f"{split_name}: Returning!")
    return df_combined


def load_train(outcome_col_name, n_to_load: Union[None, int] = None):
    return load_combined(
        "train", outcome_col_name=outcome_col_name, n_to_load=n_to_load
    )


def load_test(outcome_col_name, n_to_load: Union[None, int] = None):
    return load_combined("test", outcome_col_name=outcome_col_name, n_to_load=n_to_load)


def load_val(outcome_col_name, n_to_load: Union[None, int] = None):
    return load_combined("val", outcome_col_name=outcome_col_name, n_to_load=n_to_load)
