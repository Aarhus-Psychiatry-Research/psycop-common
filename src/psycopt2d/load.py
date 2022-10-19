"""Loader for the t2d dataset."""
import re
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from psycopmlutils.sql.loader import sql_load
from pydantic import BaseModel, Field
from wasabi import Printer

from psycopt2d.utils import PROJECT_ROOT, coerce_to_datetime

msg = Printer(timestamp=True)


class DatasetTimeSpecification(BaseModel):
    """Specification of the time range of the dataset."""

    drop_patient_if_outcome_before_date: Optional[Union[str, datetime]] = Field(
        description="""If a patient experiences the outcome before this date, all their prediction times will be dropped.
        Used for wash-in, to avoid including patients who were probably already experiencing the outcome before the study began.""",
    )

    min_prediction_time_date: Optional[Union[str, datetime]] = Field(
        description="""Any prediction time before this date will be dropped.""",
    )

    min_lookbehind_days: Optional[Union[int, float]] = Field(
        description="""If the distance from the prediction time to the start of the dataset is less than this, the prediction time will be dropped""",
    )

    min_lookahead_days: Optional[Union[int, float]] = Field(
        description="""If the distance from the prediction time to the end of the dataset is less than this, the prediction time will be dropped""",
    )

    lookbehind_combination: Optional[list[Union[int, float]]] = Field(
        description="""List containing a combination of lookbehind windows (e.g. [30, 60, 90]) which determines which features to keep in the dataset. E.g. for the above list, only features with lookbehinds of 30, 60 or 90 days will be kept.""",
    )


class DatasetSpecification(BaseModel):
    """Specification for loading a dataset."""

    split_dir_path: Union[str, Path] = Field(
        description="""Path to the directory containing the split files.""",
    )

    file_suffix: str = Field(
        description="""Suffix of the split files. E.g. 'parquet' or 'csv'.""",
        default="parquet",
    )

    time: DatasetTimeSpecification

    pred_col_name_prefix: str = Field(
        default="pred_",
        description="""Prefix for the prediction column names.""",
    )
    pred_time_colname: str = Field(
        default="timestamp",
        description="""Column name for with timestamps for prediction times""",
    )


def load_timestamp_for_any_diabetes():
    """Loads timestamps for the broad definition of diabetes used for wash-in.

    See R files for details.
    """
    timestamp_any_diabetes = sql_load(
        query="SELECT * FROM [fct].[psycop_t2d_first_diabetes_any]",
        format_timestamp_cols_to_datetime=False,
    )[["dw_ek_borger", "datotid_first_diabetes_any"]]

    timestamp_any_diabetes = timestamp_any_diabetes.rename(
        columns={"datotid_first_diabetes_any": "timestamp_washin"},
    )

    return timestamp_any_diabetes


def add_washin_timestamps(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add washin timestamps to dataset.

    Washin is an exclusion criterion. E.g. if the patient has any visit
    that looks like diabetes before the study starts (i.e. during
    washin), they are excluded.
    """
    timestamp_washin = load_timestamp_for_any_diabetes()

    dataset = dataset.merge(
        timestamp_washin,
        on="dw_ek_borger",
        how="left",
    )

    return dataset


class DataLoader:
    """Class to handle loading of a datasplit."""

    def __init__(
        self,
        spec: DatasetSpecification,
    ):
        self.spec = spec

        # File handling
        self.dir_path = Path(spec.split_dir_path)
        self.file_suffix = spec.file_suffix

        # Column specifications
        self.pred_col_name_prefix = spec.pred_col_name_prefix

    def _load_dataset_file(  # pylint: disable=inconsistent-return-statements
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

    def _drop_rows_if_datasets_ends_within_days(
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
            max_datetime = dataset[self.spec.pred_time_colname].max() - n_days
            before_max_dt = dataset[self.spec.pred_time_colname] < max_datetime
            dataset = dataset[before_max_dt]
        elif direction == "behind":
            min_datetime = dataset[self.spec.pred_time_colname].min() + n_days
            after_min_dt = dataset[self.spec.pred_time_colname] > min_datetime
            dataset = dataset[after_min_dt]

        return dataset

    def _drop_patients_with_event_in_washin(self, dataset) -> pd.DataFrame:
        """Drop patients within washin period."""

        # Remove dates before drop_patient_if_outcome_before_date
        outcome_before_date = (
            dataset["timestamp_first_diabetes_any"]
            < self.spec.time.drop_patient_if_outcome_before_date
        )

        patients_to_drop = set(dataset["dw_ek_borger"][outcome_before_date].unique())
        dataset = dataset[~dataset["dw_ek_borger"].isin(patients_to_drop)]

        # Removed dates before drop_patient_if_outcome_before_date
        dataset = dataset[
            dataset[self.spec.pred_time_colname]
            > self.spec.time.drop_patient_if_outcome_before_date
        ]

        return dataset

    def _drop_cols_not_in_lookbehind_combination(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """Drop predictor columns that are not in the specified combination of
        lookbehind windows.

        Args:
            dataset (pd.DataFrame): Dataset.

        Returns:
            pd.DataFrame: Dataset with dropped columns.
        """

        # Extract all unique lookbhehinds in the dataset predictors
        lookbehinds_in_dataset = {
            int(re.findall(r"within_(\d+)_days", col)[0])
            for col in dataset.columns
            if self.pred_col_name_prefix in col
        }

        # Check that all loobehinds in lookbehind_combination are used in the predictors
        if not set(self.spec.time.lookbehind_combination).issubset(
            lookbehinds_in_dataset,
        ):
            raise ValueError(
                f"One or more of the provided lookbehinds in lookbehind_combination is/are not used in any predictors in the dataset. Lookbehinds in dataset: {lookbehinds_in_dataset}. Lookbehinds in lookbehind_combination: {self.spec.time.lookbehind_combination}.",
            )

        # Create a list of all predictor columns who have a lookbehind window not in lookbehind_combination list
        cols_to_drop = [
            col
            for col in dataset.columns
            if any(
                str(x) not in col and self.pred_col_name_prefix in col
                for x in self.spec.time.lookbehind_combination
            )
        ]

        dataset = dataset.drop(columns=cols_to_drop)

        return dataset

    def _convert_timestamp_dtype_and_nat(self, dataset: pd.DataFrame) -> pd.DataFrame:
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

    def _drop_cols_if_exceeds_look_direction_threshold(
        self,
        dataset: pd.DataFrame,
        look_direction_threshold: Union[int, float],
        direction: str,
    ) -> pd.DataFrame:
        """Drop columns if they look behind or ahead longer than a specified
        threshold.

            For example, if direction is "ahead", and n_days is 30, then the column
        should be dropped if it's trying to look 60 days ahead. This is useful
        to avoid some rows having more information than others.

            Args:
                dataset (pd.DataFrame): Dataset to process.
                look_direction_threshold (Union[int, float]): Number of days to look in the direction.
                direction (str): Direction to look. Allowed are ["ahead", "behind"].

        Returns:
            pd.DataFrame: Dataset without the dropped columns.
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

                if lookbehind_days > look_direction_threshold:
                    cols_to_drop.append(col)

        return dataset[[c for c in dataset.columns if c not in cols_to_drop]]

    def _drop_cols_and_rows_if_look_direction_not_met(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """Drop columns if they are outside the specification. Specifically:

        - min_lookahead_days is insufficient for the column's lookahead
        - min_lookbehind_days is insufficient for the column's lookbehind
        - The dataset doesn't stretch far enough for the prediction time's lookahead
        - The dataset doesn't stretch far enough for the prediction time's lookbehind

        Args:
            dataset (pd.DataFrame): Dataset to process.
        """
        for direction in ("ahead", "behind"):

            if direction in ("ahead", "behind"):
                if self.spec.time.min_lookahead_days:
                    n_days = self.spec.time.min_lookahead_days
                elif self.spec.time.min_lookbehind_days:
                    n_days = self.spec.time.min_lookbehind_days
                else:
                    continue

            dataset = self._drop_rows_if_datasets_ends_within_days(
                n_days=n_days,
                dataset=dataset,
                direction=direction,
            )

            dataset = self._drop_cols_if_exceeds_look_direction_threshold(
                dataset=dataset,
                look_direction_threshold=n_days,
                direction=direction,
            )

        return dataset

    def _process_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Process dataset, namely:

        - Drop patients with outcome before drop_patient_if_outcome_before_date
        - Process timestamp columns
        - Drop visits where mmin_lookahead, min_lookbehind or min_prediction_time_date are not met
        - Drop features with lookbehinds not in lookbehind_combination

        Returns:
            pd.DataFrame: Processed dataset
        """
        if self.spec.time.drop_patient_if_outcome_before_date:
            dataset = add_washin_timestamps(dataset=dataset)

        dataset = self._convert_timestamp_dtype_and_nat(dataset)
        if self.spec.time.drop_patient_if_outcome_before_date:
            dataset = self._drop_patients_with_event_in_washin(dataset=dataset)

        # Drop if later than min prediction time date
        if self.spec.time.min_prediction_time_date:
            dataset = dataset[
                dataset[self.spec.pred_time_colname]
                > self.spec.time.min_prediction_time_date
            ]

        dataset = self._drop_cols_and_rows_if_look_direction_not_met(dataset=dataset)

        if self.spec.time.lookbehind_combination:
            dataset = self._drop_cols_not_in_lookbehind_combination(dataset=dataset)

        return dataset

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
        for timedelta_arg in (
            self.spec.time.min_lookbehind_days,
            self.spec.time.min_lookahead_days,
        ):
            if timedelta_arg:
                timedelta_arg = timedelta(days=timedelta_arg)  # type: ignore

        for date_arg in (
            self.spec.time.drop_patient_if_outcome_before_date,
            self.spec.time.min_prediction_time_date,
        ):
            if isinstance(date_arg, str):
                date_arg = coerce_to_datetime(
                    date_repr=date_arg,
                )

        # Concat splits if multiple are given
        if isinstance(split_names, (list, tuple)):
            if isinstance(split_names, Iterable):
                split_names = tuple(split_names)

            if nrows is not None:
                nrows = int(
                    nrows / len(split_names),
                )

            return pd.concat(
                [
                    self._load_dataset_file(split_name=split, nrows=nrows)
                    for split in split_names
                ],
                ignore_index=True,
            )
        elif isinstance(split_names, str):
            dataset = self._load_dataset_file(split_name=split_names, nrows=nrows)

        dataset = self._process_dataset(dataset=dataset)

        msg.good(f"{split_names}: Returning!")
        return dataset


def _init_spec_from_cfg(
    cfg: DictConfig,
) -> DatasetSpecification:
    """Initialise a feature spec from a DictConfig."""
    data_cfg: dict[str, Any] = OmegaConf.to_container(  # type: ignore
        cfg.data,
        resolve=True,
    )

    if data_cfg["source"] == "synthetic":
        split_dir_path = PROJECT_ROOT / "tests" / "test_data" / "synth_splits"
        file_suffix = "csv"
    else:
        split_dir_path = data_cfg["dir"]
        file_suffix = data_cfg["source"]

    time_spec = DatasetTimeSpecification(
        drop_patient_if_outcome_before_date=data_cfg[
            "drop_patient_if_outcome_before_date"
        ],
        min_lookahead_days=data_cfg["min_lookahead_days"],
        min_lookbehind_days=data_cfg["min_lookbehind_days"],
        min_prediction_time_date=data_cfg["min_prediction_time_date"],
        lookbehind_combination=data_cfg["lookbehind_combination"],
    )

    return DatasetSpecification(
        split_dir_path=split_dir_path,
        pred_col_name_prefix=data_cfg["pred_col_name_prefix"],
        file_suffix=file_suffix,
        pred_time_colname=data_cfg["pred_timestamp_col_name"],
        n_training_samples=data_cfg["n_training_samples"],
        time=time_spec,
    )


class SplitDataset(BaseModel):
    """A dataset split into train, test and optionally validation."""

    class Config:
        """Configuration for the dataclass to allow pd.DataFrame as type."""

        arbitrary_types_allowed = True

    train: pd.DataFrame
    test: Optional[pd.DataFrame] = None
    val: pd.DataFrame


def load_train_and_val_from_cfg(cfg: DictConfig):
    """Load train and validation data from file."""

    data_specification = _init_spec_from_cfg(
        cfg,
    )

    split = DataLoader(spec=data_specification)

    return SplitDataset(
        train=split.load_dataset_from_dir(split_names="train"),
        val=split.load_dataset_from_dir(split_names="val"),
    )
