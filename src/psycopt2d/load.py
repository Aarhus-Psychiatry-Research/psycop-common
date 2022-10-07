"""Loader for the t2d dataset."""
# from datetime import date, datetime, timedelta
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from psycopmlutils.sql.loader import sql_load
from wasabi import Printer

from psycopt2d.utils import coerce_to_datetime

msg = Printer(timestamp=True)


def load_dataset_file(  # pylint: disable=inconsistent-return-statements
    split_name: str,
    dir_path: Path,
    nrows: Optional[int],
    file_suffix: str = "parquet",
) -> pd.DataFrame:
    """Load dataset from directory. Finds any .csv with the split name in its
    filename.

    Args:
        split_name (str): Name of split, allowed are ["train", "test", "val"]
        dir_path (Path): Directory of the dataset.
        nrows (Optional[int]): Number of rows to load. Defaults to None, in which case
            all rows are loaded.
        file_suffix (str, optional): File suffix of the dataset. Defaults to "parquet".

    Returns:
        pd.DataFrame: The dataset
    """
    if file_suffix not in ("csv", "parquet"):
        raise ValueError(f"File suffix {file_suffix} not supported.")

    if split_name not in ("train", "test", "val"):
        raise ValueError(f"Split name {split_name} not supported.")

    # Use glob to find the file
    path = list(dir_path.glob(f"*{split_name}*.{file_suffix}"))[0]

    if "parquet" in file_suffix:
        if nrows:
            raise ValueError(
                "nrows is not supported for parquet files. Please use csv files.",
            )
        return pd.read_parquet(path)
    elif "csv" in file_suffix:
        return pd.read_csv(filepath_or_buffer=path, nrows=nrows)


def drop_if_datasets_ends_within_days(
    pred_datetime_column: str,
    n_days: Union[float, int, timedelta],
    dataset: pd.DataFrame,
    direction: str,
) -> pd.DataFrame:
    """Drop visits where the dataset ends within a certain amount of days.

    Args:
        pred_datetime_column (str): Name of the column containing the prediction
            datetime.
        n_days (Union[float, int]): Number of days.
        dataset (pd.DataFrame): Dataset.
        direction (str): Direction to look. Allowed are ["before", "after"].

    Returns:
        pd.DataFrame: Dataset with dropped rows.
    """
    if not isinstance(n_days, timedelta):
        n_days = timedelta(days=n_days)

    if direction not in ("ahead", "behind"):
        raise ValueError(f"Direction {direction} not supported.")

    if direction == "ahead":
        max_datetime = dataset[pred_datetime_column].max() - n_days
        before_max_dt = dataset[pred_datetime_column] < max_datetime
        dataset = dataset[before_max_dt]
    elif direction == "behind":
        min_datetime = dataset[pred_datetime_column].min() + n_days
        after_min_dt = dataset[pred_datetime_column] > min_datetime
        dataset = dataset[after_min_dt]

    return dataset


def drop_patients_with_event_in_washin(
    drop_patient_if_outcome_before_date: datetime,
    dataset: pd.DataFrame,
    pred_datetime_column: str,
) -> pd.DataFrame:
    """Drop patients within washin period."""

    outcome_before_date = (
        dataset["timestamp_first_diabetes_any"] < drop_patient_if_outcome_before_date
    )

    patients_to_drop = set(dataset["dw_ek_borger"][outcome_before_date].unique())
    dataset = dataset[~dataset["dw_ek_borger"].isin(patients_to_drop)]

    # Removed dates before drop_patient_if_outcome_before_date
    dataset = dataset[
        dataset[pred_datetime_column] > drop_patient_if_outcome_before_date
    ]

    return dataset


def process_timestamp_dtype_and_nat(dataset: pd.DataFrame) -> pd.DataFrame:
    """Process timestamp dtype and NaT values."""
    timestamp_colnames = [col for col in dataset.columns if "timestamp" in col]

    for colname in timestamp_colnames:
        if dataset[colname].dtype != "datetime64[ns]":
            # Convert all 0s in colname to NaT
            dataset[colname] = dataset[colname].apply(
                lambda x: pd.NaT if x == "0" else x,
            )
            dataset[colname] = pd.to_datetime(dataset[colname])

    return dataset


def add_washin_timestamps(dataset):
    """Add washin timestamps to dataset."""
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

    return dataset


def process_dataset(
    dataset: pd.DataFrame,
    drop_patient_if_outcome_before_date: datetime,
    pred_datetime_column: str,
    min_lookahead_days: Union[int, float],
    min_lookbehind_days: Union[int, float],
    min_prediction_time_date: datetime,
) -> pd.DataFrame:
    """Process dataset.

    Args:
        dataset (pd.DataFrame): Dataset to process.
        drop_patient_if_outcome_before_date (datetime): Remove patients which
            experienced an outcome prior to the date. Also removes all visits prior to
            this date as otherwise the model might learn that no visits prior to the date can be tagged with the outcome.
            Takes either a datetime or a str in isoformat (e.g. 2022-01-01). Defaults to None.
        pred_datetime_column (str): Column with prediction time timestamps.
        min_lookahead_days (Union[int, float]): Minimum amount of days from prediction time to end of dataset for the visit time to be included.
            Useful if you're looking e.g. 5 years ahead for your outcome, but some visits only have 1 year of lookahead.
            Defined as days from the last days.
        min_lookbehind_days (Union[int, float]): Minimum amount of days from prediction time to start of dataset for the visit time to be included.
        min_prediction_time_date (datetime): Minimum prediction time date. Defaults to None.

    Returns:
        pd.DataFrame: Processed dataset
    """
    if drop_patient_if_outcome_before_date:
        dataset = add_washin_timestamps(dataset=dataset)

    dataset = process_timestamp_dtype_and_nat(dataset)

    if drop_patient_if_outcome_before_date:
        dataset = drop_patients_with_event_in_washin(
            dataset=dataset,
            drop_patient_if_outcome_before_date=drop_patient_if_outcome_before_date,
            pred_datetime_column=pred_datetime_column,
        )

    # Drop if later than min prediction time date
    if min_prediction_time_date:
        dataset = dataset[dataset[pred_datetime_column] > min_prediction_time_date]

    for direction in ("ahead", "behind"):
        n_days = min_lookahead_days if direction == "ahead" else min_lookbehind_days

        dataset = drop_if_datasets_ends_within_days(
            dataset=dataset,
            pred_datetime_column=pred_datetime_column,
            n_days=n_days,
            direction=direction,
        )

    return dataset


def load_dataset_from_dir(
    split_names: Union[Iterable[str], str],
    dir_path: Path,
    drop_patient_if_outcome_before_date: datetime,
    min_lookahead_days: Union[float, int],
    min_lookbehind_days: Union[float, int],
    min_prediction_time_date: Union[str, datetime],
    file_suffix: str = "parquet",
    pred_datetime_column: str = "timestamp",
    n_training_samples: Union[None, int] = None,
) -> pd.DataFrame:
    """Load dataset for t2d.

    Args:
        split_names (Union[Iterable[str], str]): Names of splits, includes "train", "val",
            "test".
        dir_path (Path): Directory of the dataset.
        drop_patient_if_outcome_before_date (Union[datetime, str]): Remove patients which
            experienced an outcome prior to the date. Also removes all visits prior to
            this date as otherwise the model might learn that no visits prior to the date can be tagged with the outcome.
            Takes either a datetime or a str in isoformat (e.g. 2022-01-01). Defaults to None.
        min_lookahead_days (int): Minimum amount of days from prediction time to end of dataset for the visit time to be included.
            Useful if you're looking e.g. 5 years ahead for your outcome, but some visits only have 1 year of lookahead.
            Defined as days from the last days.
        min_lookbehind_days (int): Minimum amount of days from prediction time to start of dataset for the visit time to be included.
        min_prediction_time_date (Union[str, datetime]): Minimum date for a prediction time to be included in the dataset.
        file_suffix (str): File suffix of the dataset. Defaults to "parquet".
        pred_datetime_column (str, optional): Column with prediction time timestamps.
            Defaults to "timestamp".
        n_training_samples (Union[None, int], optional): Number of training samples to load.
        Defaults to None, in which case all training samples are loaded.

    Returns:
        pd.DataFrame: The filtered dataset
    """
    # Handle input types
    for timedelta_arg in (min_lookbehind_days, min_lookahead_days):
        timedelta_arg = timedelta(days=timedelta_arg)  # type: ignore

    for date_arg in (drop_patient_if_outcome_before_date, min_prediction_time_date):
        if isinstance(date_arg, str):
            date_arg = coerce_to_datetime(
                date_repr=date_arg,
            )

    # Concat splits if multiple are given
    if isinstance(split_names, (list, tuple)):

        if isinstance(split_names, Iterable):
            split_names = tuple(split_names)

        if n_training_samples is not None:
            n_training_samples = int(n_training_samples / len(split_names))

        return pd.concat(
            [
                load_dataset_file(
                    split_name=split,
                    dir_path=dir_path,
                    nrows=n_training_samples,
                    file_suffix=file_suffix,
                )
                for split in split_names
            ],
            ignore_index=True,
        )
    elif isinstance(split_names, str):
        dataset = load_dataset_file(
            split_name=split_names,
            dir_path=dir_path,
            nrows=n_training_samples,
            file_suffix=file_suffix,
        )

    dataset = process_dataset(
        dataset=dataset,
        drop_patient_if_outcome_before_date=drop_patient_if_outcome_before_date,
        pred_datetime_column=pred_datetime_column,
        min_lookahead_days=min_lookahead_days,
        min_lookbehind_days=min_lookbehind_days,
        min_prediction_time_date=min_prediction_time_date,
    )

    msg.good(f"{split_names}: Returning!")
    return dataset
