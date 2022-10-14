"""Loader for the t2d dataset."""
# from datetime import date, datetime, timedelta
import re
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Sized, Union

import omegaconf
import pandas as pd
from omegaconf import open_dict
from psycopmlutils.sql.loader import sql_load
from sklearn.model_selection import train_test_split
from wasabi import Printer

from psycopt2d.utils import PROJECT_ROOT, coerce_to_datetime

msg = Printer(timestamp=True)


def load_dataset_file(  # pylint: disable=inconsistent-return-statements
    split_name: str,
    dir_path: Path,
    nrows: Optional[int],
    file_suffix: str = "parquet",
) -> pd.DataFrame:
    """Load dataset from directory. Finds any file with the matching file
    suffix with the split name in its filename.

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

    path = list(dir_path.glob(f"*{split_name}*.{file_suffix}"))[0]

    if "parquet" in file_suffix:
        if nrows:
            raise ValueError(
                "nrows is not supported for parquet files. Please use csv files.",
            )
        return pd.read_parquet(path)
    elif "csv" in file_suffix:
        return pd.read_csv(filepath_or_buffer=path, nrows=nrows)


def drop_rows_if_datasets_ends_within_days(
    pred_datetime_column: str,
    n_days: Union[float, int, timedelta],
    dataset: pd.DataFrame,
    direction: str,
) -> pd.DataFrame:
    """Drop visits that lie within certain amount of days from end of dataset.

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
    """Convert columns with `timestamp`in their name to datetime, and convert
    0's to NaT."""
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


def drop_columns_if_min_look_direction_not_met(
    dataset: pd.DataFrame,
    n_days: Union[int, float],
    direction: str,
    pred_col_name_prefix: str,
) -> pd.DataFrame:
    """Drop columns if the minimum look direction is not met.

    Args:
        dataset (pd.DataFrame): Dataset to process.
        n_days (Union[int, float]): Number of days to look in the direction.
        direction (str): Direction to look. Allowed are ["ahead", "behind"].
        pred_col_name_prefix (str): Prefix of the prediction column names.

    Returns:
        pd.DataFrame: Dataset with dropped columns.
    """
    cols_to_drop = []

    if direction == "behind":
        cols_to_process = [c for c in dataset.columns if pred_col_name_prefix in c]

        for col in cols_to_process:
            # Extract lookbehind days from column name use regex
            # E.g. "column_name_within_90_days" == 90
            # E.g. "column_name_within_90_days_fallback_NaN" == 90
            lookbehind_days_strs = re.findall(r"within_(\d+)_days", col)

            if len(lookbehind_days_strs) > 0:
                lookbehind_days = int(lookbehind_days_strs[0])
            else:
                raise ValueError(f"Could not extract lookbehind days from {col}")

            if lookbehind_days > n_days:
                cols_to_drop.append(col)

    return dataset[[c for c in dataset.columns if c not in cols_to_drop]]


def process_dataset(
    dataset: pd.DataFrame,
    drop_patient_if_outcome_before_date: datetime,
    pred_datetime_column: str,
    min_lookahead_days: Union[int, float],
    min_lookbehind_days: Union[int, float],
    min_prediction_time_date: datetime,
    pred_col_name_prefix: str,
) -> pd.DataFrame:
    """Process dataset, namely:

    - Drop patients with outcome before drop_patient_if_outcome_before_date
    - Process timestamp columns
    - Drop visits where mmin_lookahead, min_lookbehind or min_prediction_time_date are not met

    And return the dataset.

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
        pred_col_name_prefix (str): Prefix of prediction columns.

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

        dataset = drop_rows_if_datasets_ends_within_days(
            dataset=dataset,
            pred_datetime_column=pred_datetime_column,
            n_days=n_days,
            direction=direction,
        )

        dataset = drop_columns_if_min_look_direction_not_met(
            dataset=dataset,
            n_days=n_days,
            direction=direction,
            pred_col_name_prefix=pred_col_name_prefix,
        )

    return dataset


def load_dataset_from_dir(
    split_names: Union[Iterable[str], str],
    dir_path: Path,
    drop_patient_if_outcome_before_date: datetime,
    min_lookahead_days: Union[float, int],
    min_lookbehind_days: Union[float, int],
    min_prediction_time_date: datetime,
    pred_col_name_prefix: str,
    file_suffix: str = "parquet",
    pred_datetime_column: str = "timestamp",
    n_training_samples: Union[None, int] = None,
) -> pd.DataFrame:
    """Load dataset for t2d. Can load multiple splits at once, e.g. concatenate
    train and val for crossvalidation.

    Args:
        split_names (Union[Iterable[str], str]): Names of splits, includes "train", "val",
            "test". Can take multiple splits and concatenate them for crossvalidation.
        dir_path (Path): Directory of the dataset.
        drop_patient_if_outcome_before_date (datetime): Remove patients which
            experienced an outcome prior to the date. Also removes all visits prior to
            this date as otherwise the model might learn that no visits prior to the date can be tagged with the outcome.
            Takes either a datetime or a str in isoformat (e.g. 2022-01-01). Defaults to None.
        min_lookahead_days (int): Minimum amount of days from prediction time to end of dataset for the visit time to be included.
            Useful if you're looking e.g. 5 years ahead for your outcome, but some visits only have 1 year of lookahead.
            Defined as days from the last days.
        min_lookbehind_days (int): Minimum amount of days from prediction time to start of dataset for the visit time to be included.
        min_prediction_time_date (Union[str, datetime]): Minimum date for a prediction time to be included in the dataset.
        pred_col_name_prefix (str): Prefix of prediction columns. Defaults to "pred_".
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
        pred_col_name_prefix=pred_col_name_prefix,
    )

    msg.good(f"{split_names}: Returning!")
    return dataset


def load_synth_train_val_from_dir(cfg, synth_splits_dir):
    """Load synthetic train and val data from dir."""
    # This is a temp fix. We probably want to use pydantic to validate all our inputs, and to set defaults if they don't exist in the config
    try:
        type(cfg.data.drop_patient_if_outcome_before_date)
    except omegaconf.errors.ConfigAttributeError:
        # Assign as none to struct
        with open_dict(cfg) as conf_dict:
            conf_dict.data.drop_patient_if_outcome_before_date = None

    train = load_dataset_from_dir(
        split_names="train",
        dir_path=synth_splits_dir,
        n_training_samples=cfg.data.n_training_samples,
        drop_patient_if_outcome_before_date=cfg.data.drop_patient_if_outcome_before_date,
        min_lookahead_days=cfg.data.min_lookahead_days,
        min_lookbehind_days=cfg.data.min_lookbehind_days,
        min_prediction_time_date=cfg.data.min_prediction_time_date,
        file_suffix="csv",
    )

    val = load_dataset_from_dir(
        split_names="val",
        dir_path=synth_splits_dir,
        n_training_samples=cfg.data.n_training_samples,
        drop_patient_if_outcome_before_date=cfg.data.drop_patient_if_outcome_before_date,
        min_lookahead_days=cfg.data.min_lookahead_days,
        min_lookbehind_days=cfg.data.min_lookbehind_days,
        min_prediction_time_date=cfg.data.min_prediction_time_date,
        file_suffix="csv",
    )

    return train, val


def gen_synth_data_splits(cfg, test_data_dir):
    """Generate synthetic data splits."""
    dataset = pd.read_csv(
        test_data_dir / "synth_prediction_data.csv",
    )

    # Convert all timestamp cols to datetime
    for col in [col for col in dataset.columns if "timestamp" in col]:
        dataset[col] = pd.to_datetime(dataset[col])

    # Get 75% of dataset for train
    train, val = train_test_split(
        dataset,
        test_size=0.25,
        random_state=cfg.project.seed,
    )

    return train, val


def write_synth_splits(  # pylint: disable=unused-argument
    test_data_dir: Path,
    train,
    val,
):
    """Write synthetic data splits to disk."""

    synth_splits_dir = test_data_dir / "synth_splits"
    synth_splits_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        split_df = eval(split)  # pylint: disable=eval-used
        split_df.to_csv(synth_splits_dir / f"{split}.csv", index=False)

    return synth_splits_dir


def load_train_and_val_from_file(cfg):
    """Load data from file."""
    if cfg.data.source in ("csv", "parquet"):
        path = Path(cfg.data.dir)
        file_suffix = cfg.data.source
    elif cfg.data.source == "synthetic":
        path = PROJECT_ROOT / "tests" / "test_data" / "synth_splits"
        file_suffix = "csv"
    else:
        raise ValueError(f"Unknown data source: {cfg.data.source}")

    train = load_dataset_from_dir(
        split_names="train",
        dir_path=path,
        n_training_samples=cfg.data.n_training_samples,
        drop_patient_if_outcome_before_date=cfg.data.drop_patient_if_outcome_before_date,
        min_lookahead_days=cfg.data.min_lookahead_days,
        min_lookbehind_days=cfg.data.min_lookbehind_days,
        min_prediction_time_date=cfg.data.min_prediction_time_date,
        file_suffix=file_suffix,
        pred_col_name_prefix=cfg.data.pred_col_name_prefix,
    )

    val = load_dataset_from_dir(
        split_names="val",
        dir_path=path,
        n_training_samples=cfg.data.n_training_samples,
        drop_patient_if_outcome_before_date=cfg.data.drop_patient_if_outcome_before_date,
        min_lookahead_days=cfg.data.min_lookahead_days,
        min_lookbehind_days=cfg.data.min_lookbehind_days,
        min_prediction_time_date=cfg.data.min_prediction_time_date,
        file_suffix=file_suffix,
        pred_col_name_prefix=cfg.data.pred_col_name_prefix,
    )

    return train, val


def load_dataset_with_config(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset based on settings in the config file."""

    allowed_data_sources = {"csv", "parquet", "synthetic"}

    if "csv" in cfg.data.source.lower() or "parquet" in cfg.data.source.lower():
        train, val = load_train_and_val_from_file(cfg)

    elif cfg.data.source.lower() == "synthetic":
        train, val = load_train_and_val_from_file(cfg)

    else:
        raise ValueError(
            f"The config data.source is {cfg.data.source}, allowed are {allowed_data_sources}",
        )

    return train, val
