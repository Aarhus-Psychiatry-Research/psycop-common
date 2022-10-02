# from datetime import date, datetime, timedelta
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Union, list

import pandas as pd
from psycopmlutils.sql.loader import sql_load
from wasabi import Printer

msg = Printer(timestamp=True)


def load_dataset(
    split_names: Union[list[str], str],
    dir: Path,
    drop_patient_if_outcome_before_date: Union[datetime, str],
    min_lookahead_days: int,
    pred_datetime_column: Optional[str] = "timestamp",
    n_training_samples: Union[None, int] = None,
) -> pd.DataFrame:
    """Load dataset for t2d.

    Args:
        split_names (Union[list[str], str]): Names of splits, includes "train", "val",
            "test".
        dir (Path): Directory of the dataset.
        drop_patient_if_outcome_before_date (Union[datetime, str]): Remove patients which
            experienced an outcome prior to the date. Also removes all visits prior to
            this date as otherwise the model might learn that no visits prior to the date can be tagged with the outcome.
            Takes either a datetime or a str in isoformat (e.g. 2022-01-01). Defaults to None.
        min_lookahead_days (int): Minimum amount of days from prediction time to end of dataset for the visit time to be included.
            Useful if you're looking e.g. 5 years ahead for your outcome, but some visits only have 1 year of lookahead.
            Defined as days from the last days.
        pred_datetime_column (str, optional): Column with prediction time timestamps.
        Defaults to "timestamp".
        n_training_samples (Union[None, int], optional): Number of training samples to load.
        Defaults to None, in which case all training samples are loaded.

    Returns:
        pd.DataFrame: The filtered dataset
    """
    min_lookahead = timedelta(days=min_lookahead_days)

    if isinstance(drop_patient_if_outcome_before_date, str):
        drop_patient_if_outcome_before_date = date.fromisoformat(
            drop_patient_if_outcome_before_date,
        )

    # Convert drop_patient_if_outcome_before_date from a date to a datetime at midnight
    if isinstance(drop_patient_if_outcome_before_date, date):
        drop_patient_if_outcome_before_date = datetime.combine(
            drop_patient_if_outcome_before_date,
            datetime.min.time(),
        )

    if isinstance(split_names, list):
        if n_training_samples is not None:
            n_training_samples = int(n_training_samples / len(split_names))

        return pd.concat(
            [
                load_dataset(
                    split_names=split,
                    dir=dir,
                    drop_patient_if_outcome_before_date=drop_patient_if_outcome_before_date,
                    min_lookahead_days=min_lookahead_days,
                    pred_datetime_column=pred_datetime_column,
                    n_training_samples=n_training_samples,
                )
                for split in split_names
            ],
            ignore_index=True,
        )

    # Load dataset from csv
    dataset = load_dataset_from_dir(
        split_name=split_names,
        dir=dir,
        nrows=n_training_samples,
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

    outcome_before_date = (
        dataset["timestamp_first_diabetes_any"] < drop_patient_if_outcome_before_date
    )
    patients_to_drop = set(dataset["dw_ek_borger"][outcome_before_date].unique())
    dataset = dataset[~dataset["dw_ek_borger"].isin(patients_to_drop)]

    # Removed dates before drop_patient_if_outcome_before_date
    dataset = dataset[
        dataset[pred_datetime_column] > drop_patient_if_outcome_before_date
    ]

    # remove dates min_lookahead_days before last recorded timestep
    max_datetime = dataset[pred_datetime_column].max() - min_lookahead
    before_max_dt = dataset[pred_datetime_column] < max_datetime
    dataset = dataset[before_max_dt]

    msg.good(f"{split_names}: Returning!")
    return dataset


def load_dataset_from_dir(
    split_name: str,
    dir: Path,
    nrows: Optional[int],
) -> pd.DataFrame:
    """Load dataset from directory. Finds any .csv with the split name in its
    filename.

    Args:
        split_name (str): Name of split, allowed are ["train", "test", "val"]
        dir (Path): Directory of the dataset.
        nrows (Optional[int]): Number of rows to load. Defaults to None, in which case
            all rows are loaded.

    Returns:
        pd.DataFrame: The dataset
    """
    # Use glob to find the file
    path = list(dir.glob(f"*{split_name}*.csv"))[0]

    return pd.read_csv(filepath_or_buffer=path, nrows=nrows)
