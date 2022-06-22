from typing import Tuple, Union

import pandas as pd
from psycopmlutils.loaders import sql_load
from wasabi import Printer

msg = Printer(timestamp=True)


def load_dataset(
    split_name: str,
    outcome_col_name: str,
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

    # Add "any diabetes" column for wash-in
    timestamp_any_diabetes = sql_load(
        query=f"SELECT * FROM [fct].[psycop_t2d_first_diabetes_any]",
        format_timestamp_cols_to_datetime=False,
    )[["dw_ek_borger", "datotid_first_diabetes_any"]]

    timestamp_any_diabetes = timestamp_any_diabetes.rename(
        columns={"datotid_first_diabetes_any": "timestamp_first_diabetes_any"}
    )

    df_combined = df_combined.merge(
        timestamp_any_diabetes, on="dw_ek_borger", how="left"
    )

    # Convert all timestamp cols to datetime64[ns]
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

    msg.good(f"{split_name}: Returning!")
    return _X, _y
