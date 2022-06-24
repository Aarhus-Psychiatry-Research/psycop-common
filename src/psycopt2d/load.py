from datetime import date, datetime, timedelta
from typing import List, Union

import pandas as pd
from psycopmlutils.loaders import sql_load
from wasabi import Printer

msg = Printer(timestamp=True)


def load_dataset(
    split_names: Union[List[str], str],
    drop_if_outcome_before_date: Union[datetime, str],
    min_lookahead_days: int,
    datetime_column: str = "pred_timestamp",
    n_training_samples: Union[None, int] = None,
) -> pd.DataFrame:
    """load dataset for t2d.

    Args:
        split_names (Union[List[str], str]): Names of splits, includes "train", "val",
            "test".
        drop_if_outcome_before_date (Union[datetime, str]): Remove patient which have
            had their outcome before the specified day. Also removed all visits before
            this date as otherwise the model will learn that all visit before this date
            does not lead to diabetes.
            limit the dataset to. Defaults to None.
        min_lookahead_days (int): Minimum amount of days to required for lookahead.
            Defined as days from the last days.
        datetime_column (str, optional): Datetime columns. Defaults to
            "pred_timestamp".
        n_training_samples (Union[None, int], optional): Number of training samples to
    Returns:
        pd.DataFrame: The filtered dataset
    """
    if isinstance(drop_if_outcome_before_date, str):
        drop_if_outcome_before_date = date.fromisoformat(drop_if_outcome_before_date)

    if isinstance(split_names, list):
        return pd.concat(
            [
                load_dataset(
                    split,
                    drop_if_outcome_before_date,
                    min_lookahead_days,
                    datetime_column,
                    n_training_samples,
                )
                for split in split_names
            ],
        )

    min_lookahead = timedelta(days=min_lookahead_days)
    sql_table_name = f"psycop_t2d_{split_names}"

    if n_training_samples is not None:
        msg.info(f"{sql_table_name}: Loading {n_training_samples} rows from")
        select = f"SELECT TOP {n_training_samples}"
    else:
        msg.info(f"{sql_table_name}: Loading all rows")
        select = "SELECT"

    dataset = sql_load(
        query=f"{select} * FROM [fct].[{sql_table_name}]",
        format_timestamp_cols_to_datetime=False,
    )

    # Add "any diabetes" column for wash-in
    timestamp_any_diabetes = sql_load(
        query="SELECT * FROM [fct].[psycop_t2d_first_diabetes_any]",
        format_timestamp_cols_to_datetime=False,
    )[["dw_ek_borger", "datotid_first_diabetes_any"]]

    timestamp_any_diabetes = timestamp_any_diabetes.rename(
        columns={"datotid_first_diabetes_any": "timestamp_first_diabetes_any"},
    )

    dataset = dataset.merge(
        timestamp_any_diabetes,
        on="dw_ek_borger",
        how="left",
    )

    # Convert all timestamp cols to datetime64[ns]
    timestamp_colnames = [col for col in dataset.columns if "timestamp" in col]

    for colname in timestamp_colnames:
        if dataset[colname].dtype != "datetime64[ns]":
            # Convert all 0s in colname to NaT
            dataset[colname] = dataset[colname].apply(
                lambda x: pd.NaT if x == "0" else x,
            )
            dataset[colname] = pd.to_datetime(dataset[colname])

    outcome_before_date = dataset["timestamp_t2d_diag"] < drop_if_outcome_before_date
    patients_to_drop = set(dataset["dw_ek_borger"][outcome_before_date].unique())
    dataset = dataset[dataset["dw_ek_borger"].isin(patients_to_drop)]

    # Removed dates before min_datetime
    dates = dataset[datetime_column]
    above_dt = dates > drop_if_outcome_before_date
    dataset = dataset[above_dt]

    # remove dates min_lookahead_days before last recorded timestep
    max_datetime = dates.max() - min_lookahead
    below_dt = dates < max_datetime
    dataset = dataset[below_dt]

    msg.good(f"{split_names}: Returning!")
    return dataset
