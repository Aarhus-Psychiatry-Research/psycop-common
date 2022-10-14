"""Loader for the t2d dataset."""
# from datetime import date, datetime, timedelta
import re
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from omegaconf import DictConfig
from psycopmlutils.sql.loader import sql_load
from pydantic import BaseModel
from wasabi import Printer

from psycopt2d.utils import PROJECT_ROOT, coerce_to_datetime

msg = Printer(timestamp=True)


class DatasetSpec(BaseModel):
    """Specification for loading a dataset."""

    split_names: Union[str, Iterable[str]]
    split_dir_path: Union[str, Path]
    drop_patient_if_outcome_before_date: Optional[Union[str, datetime]] = None
    min_prediction_time_date: Optional[Union[str, datetime]] = None
    pred_col_name_prefix: str = "pred_"
    file_suffix: str = "parquet"
    pred_time_colname: str = "timestamp"
    n_training_samples: Optional[int] = None
    min_lookahead_days: Optional[Union[int, float]] = None
    min_lookbehind_days: Optional[Union[int, float]] = None


def add_washin_timestamps(dataset: pd.DataFrame) -> pd.DataFrame:
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


class DataSplit:
    """Class to handle loading of a datasplit."""

    def __init__(
        self,
        spec: DatasetSpec,
    ):
        # Init all the args
        self.split_names = spec.split_names
        self.pred_time_colname = spec.pred_time_colname

        # File handling
        self.dir_path = spec.split_dir_path
        self.file_suffix = spec.file_suffix

        # Column specifications
        self.pred_col_name_prefix = spec.pred_col_name_prefix

        # How much data to load
        self.n_training_samples = spec.n_training_samples

        # Computation
        self.train_samples = (
            int(self.n_training_samples * 0.7) if self.n_training_samples else None
        )
        self.val_samples = (
            int(self.n_training_samples * 0.3) if self.n_training_samples else None
        )

        # Ahead
        self.min_lookahead_days = spec.min_lookahead_days

        # Behind
        self.drop_patient_if_outcome_before_date = (
            spec.drop_patient_if_outcome_before_date
        )
        self.min_prediction_time_date = spec.min_prediction_time_date
        self.min_lookbehind_days = spec.min_lookbehind_days

        self.df = self.load_dataset_from_dir(
            split_names=self.split_names,
            nrows=self.n_training_samples,
        )

    def load_dataset_from_dir(
        self,
        split_names: Union[Iterable[str], str],
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load dataset for t2d. Can load multiple splits at once, e.g.
        concatenate train and val for crossvalidation.

        Args:
            split_names (Union[Iterable[str], str]): Name of split, allowed are ["train", "test", "val"]
            nrows (Optional[int]): Number of rows to load from dataset. Defaults to None, in which case all rows are loaded.

        Returns:
            pd.DataFrame: The filtered dataset
        """
        # Handle input types
        for timedelta_arg in (self.min_lookbehind_days, self.min_lookahead_days):
            if timedelta_arg:
                timedelta_arg = timedelta(days=timedelta_arg)  # type: ignore

        for date_arg in (
            self.drop_patient_if_outcome_before_date,
            self.min_prediction_time_date,
        ):
            if isinstance(date_arg, str):
                date_arg = coerce_to_datetime(
                    date_repr=date_arg,
                )

        # Concat splits if multiple are given
        if isinstance(split_names, (list, tuple)):
            if isinstance(split_names, Iterable):
                split_names = tuple(split_names)

            if self.n_training_samples is not None:
                self.n_training_samples = int(
                    self.n_training_samples / len(split_names),
                )

            return pd.concat(
                [
                    self.load_dataset_file(split_name=split, nrows=nrows)
                    for split in split_names
                ],
                ignore_index=True,
            )
        elif isinstance(split_names, str):
            dataset = self.load_dataset_file(split_name=split_names, nrows=nrows)

        dataset = self.process_dataset(dataset=dataset)

        msg.good(f"{split_names}: Returning!")
        return dataset

    def load_dataset_file(  # pylint: disable=inconsistent-return-statements
        self,
        split_name: str,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:  # pylint: disable=inconsistent-return-statements
        """Load dataset from directory. Finds any file with the matching file
        suffix with the split name in its filename.

        Args:
            split_name (str): Name of split, allowed are ["train", "test", "val"]
            nrows (Optional[int]): Number of rows to load. Defaults to None, in which case
                all rows are loaded.
            self.file_suffix (str, optional): File suffix of the dataset. Defaults to "parquet".

        Returns:
            pd.DataFrame: The dataset
        """
        if self.file_suffix not in ("csv", "parquet"):
            raise ValueError(f"File suffix {self.file_suffix} not supported.")

        if split_name not in ("train", "test", "val"):
            raise ValueError(f"Split name {split_name} not supported.")

        path = list(self.dir_path.glob(f"*{split_name}*.{self.file_suffix}"))[0]

        if "parquet" in self.file_suffix:
            if nrows:
                raise ValueError(
                    "nrows is not supported for parquet files. Please use csv files.",
                )
            return pd.read_parquet(path)
        elif "csv" in self.file_suffix:
            return pd.read_csv(filepath_or_buffer=path, nrows=nrows)

    def drop_rows_if_datasets_ends_within_days(
        self,
        n_days: Union[int, float],
        dataset: pd.DataFrame,
        direction: str,
    ) -> pd.DataFrame:
        """Drop visits that lie within certain amount of days from end of
        dataset.

        Args:
            n_days (Union[float, int]): Number of days.
            dataset (pd.DataFrame): Dataset.
            direction (str): Direction to look. Allowed are ["before", "after"].

        Returns:
            pd.DataFrame: Dataset with dropped rows.
        """
        if not isinstance(n_days, timedelta):
            n_days = timedelta(days=n_days)  # type: ignore

        if direction not in ("ahead", "behind"):
            raise ValueError(f"Direction {direction} not supported.")

        if direction == "ahead":
            max_datetime = dataset[self.pred_time_colname].max() - n_days
            before_max_dt = dataset[self.pred_time_colname] < max_datetime
            dataset = dataset[before_max_dt]
        elif direction == "behind":
            min_datetime = dataset[self.pred_time_colname].min() + n_days
            after_min_dt = dataset[self.pred_time_colname] > min_datetime
            dataset = dataset[after_min_dt]

        return dataset

    def drop_patients_with_event_in_washin(self, dataset) -> pd.DataFrame:
        """Drop patients within washin period."""

        outcome_before_date = (
            dataset["timestamp_first_diabetes_any"]
            < self.drop_patient_if_outcome_before_date
        )

        patients_to_drop = set(dataset["dw_ek_borger"][outcome_before_date].unique())
        dataset = dataset[~dataset["dw_ek_borger"].isin(patients_to_drop)]

        # Removed dates before drop_patient_if_outcome_before_date
        dataset = dataset[
            dataset[self.pred_time_colname] > self.drop_patient_if_outcome_before_date
        ]

        return dataset

    def process_timestamp_dtype_and_nat(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Convert columns with `timestamp`in their name to datetime, and
        convert 0's to NaT."""
        timestamp_colnames = [col for col in dataset.columns if "timestamp" in col]

        for colname in timestamp_colnames:
            if dataset[colname].dtype != "datetime64[ns]":
                # Convert all 0s in colname to NaT
                dataset[colname] = dataset[colname].apply(
                    lambda x: pd.NaT if x == "0" else x,
                )
                dataset[colname] = pd.to_datetime(dataset[colname])

        return dataset

    def drop_columns_if_min_look_direction_not_met(
        self,
        dataset: pd.DataFrame,
        n_days: Union[int, float],
        direction: str,
    ) -> pd.DataFrame:
        """Drop columns if the minimum look direction is not met.

        Args:
            dataset (pd.DataFrame): Dataset to process.
            n_days (Union[int, float]): Number of days to look in the direction.
            direction (str): Direction to look. Allowed are ["ahead", "behind"].

        Returns:
            pd.DataFrame: Dataset with dropped columns.
        """
        cols_to_drop = []

        if direction == "behind":
            cols_to_process = [
                c for c in dataset.columns if self.pred_col_name_prefix in c
            ]

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

    def process_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Process dataset, namely:

        - Drop patients with outcome before drop_patient_if_outcome_before_date
        - Process timestamp columns
        - Drop visits where mmin_lookahead, min_lookbehind or min_prediction_time_date are not met

        Returns:
            pd.DataFrame: Processed dataset
        """
        if self.drop_patient_if_outcome_before_date:
            dataset = add_washin_timestamps(dataset=dataset)

        dataset = self.process_timestamp_dtype_and_nat(dataset)
        if self.drop_patient_if_outcome_before_date:
            dataset = self.drop_patients_with_event_in_washin(dataset=dataset)

        # Drop if later than min prediction time date
        if self.min_prediction_time_date:
            dataset = dataset[
                dataset[self.pred_time_colname] > self.min_prediction_time_date
            ]

        for direction in ("ahead", "behind"):
            if direction == "ahead":
                if self.min_lookahead_days:
                    n_days = self.min_lookahead_days
                else:
                    continue

            if direction == "behind":
                if self.min_lookbehind_days:
                    n_days = self.min_lookbehind_days
                else:
                    continue

            dataset = self.drop_rows_if_datasets_ends_within_days(
                n_days=n_days,
                dataset=dataset,
                direction=direction,
            )

            dataset = self.drop_columns_if_min_look_direction_not_met(
                dataset=dataset,
                n_days=n_days,
                direction=direction,
            )

        return dataset


def init_spec_from_cfg(cfg: DictConfig, split_name: str = "train") -> DatasetSpec:
    """Initialise a feature spec from a DictConfig."""

    data_cfg = cfg.data

    if data_cfg.source == "synthetic":
        split_dir_path = PROJECT_ROOT / "tests" / "test_data" / "synth_splits"
        file_suffix = "csv"
    else:
        split_dir_path = data_cfg.dir
        file_suffix = cfg.data.source

    return DatasetSpec(
        split_names=split_name,
        split_dir_path=split_dir_path,
        drop_patient_if_outcome_before_date=data_cfg.drop_patient_if_outcome_before_date,
        min_prediction_time_date=data_cfg.min_prediction_time_date,
        pred_col_name_prefix=data_cfg.pred_col_name_prefix,
        file_suffix=file_suffix,
        pred_time_colname=data_cfg.pred_timestamp_col_name,
        n_training_samples=data_cfg.n_training_samples,
        min_lookahead_days=data_cfg.min_lookahead_days,
        min_lookbehind_days=data_cfg.min_lookbehind_days,
    )


def load_train_and_val_from_cfg(cfg: DictConfig):
    """Load train and validation data from file."""

    train_spec = init_spec_from_cfg(cfg, split_name="train")
    val_spec = init_spec_from_cfg(cfg, split_name="val")

    return DataSplit(spec=train_spec).df, DataSplit(spec=val_spec).df
