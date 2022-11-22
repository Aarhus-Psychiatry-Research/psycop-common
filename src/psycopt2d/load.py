"""Loader for the t2d dataset."""
import os
import re
from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from psycopmlutils.sql.loader import sql_load
from pydantic import BaseModel
from wasabi import Printer

from psycopt2d.evaluate_saved_model_predictions import infer_look_distance
from psycopt2d.utils.config_schemas import FullConfigSchema
from psycopt2d.utils.pd_cache_decorator import cache_pandas_result
from psycopt2d.utils.utils import (
    FEATURE_SETS_PATH,
    get_percent_lost,
    infer_outcome_col_name,
    infer_predictor_col_name,
)

msg = Printer(timestamp=True)


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
        cfg: FullConfigSchema,
    ):
        self.cfg: FullConfigSchema = cfg

        # File handling
        self.dir_path = Path(cfg.data.dir)
        self.file_suffix = cfg.data.suffix

        # Column specifications
        self.pred_col_name_prefix = cfg.data.pred_prefix

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
            n_days_timedelt: timedelta = timedelta(days=n_days)  # type: ignore

        if direction not in ("ahead", "behind"):
            raise ValueError(f"Direction {direction} not supported.")

        n_rows_before_modification = dataset.shape[0]

        if direction == "ahead":
            max_datetime = (
                dataset[self.cfg.data.col_name.pred_timestamp].max() - n_days_timedelt
            )
            before_max_dt = (
                dataset[self.cfg.data.col_name.pred_timestamp] < max_datetime
            )
            dataset = dataset[before_max_dt]
        elif direction == "behind":
            min_datetime = (
                dataset[self.cfg.data.col_name.pred_timestamp].min() + n_days_timedelt
            )
            after_min_dt = dataset[self.cfg.data.col_name.pred_timestamp] > min_datetime
            dataset = dataset[after_min_dt]

        n_rows_after_modification = dataset.shape[0]
        percent_dropped = get_percent_lost(
            n_before=n_rows_before_modification,
            n_after=n_rows_after_modification,
        )

        if n_rows_before_modification - n_rows_after_modification != 0:
            msg.info(
                f"Dropped {n_rows_before_modification - n_rows_after_modification} ({percent_dropped}%) rows because the end of the dataset was within {n_days} of their prediction time when looking {direction} from their prediction time",
            )

        return dataset

    def _drop_patient_if_excluded(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """Drop patients that have an exclusion event within the washin
        period."""

        n_rows_before_modification = dataset.shape[0]

        outcome_before_date = (
            dataset[self.cfg.data.col_name.exclusion_timestamp]
            < self.cfg.data.drop_patient_if_exclusion_before_date
        )

        patients_to_drop = set(
            dataset[self.cfg.data.col_name.id][outcome_before_date].unique(),
        )
        dataset = dataset[~dataset[self.cfg.data.col_name.id].isin(patients_to_drop)]

        n_rows_after_modification = dataset.shape[0]

        percent_dropped = get_percent_lost(
            n_before=n_rows_after_modification,
            n_after=n_rows_after_modification,
        )

        if n_rows_before_modification - n_rows_after_modification != 0:
            msg.info(
                f"Dropped {n_rows_before_modification - n_rows_after_modification} ({percent_dropped}%) rows because they met exclusion criteria before {self.cfg.data.drop_patient_if_exclusion_before_date}.",
            )
        else:
            msg.info(
                f"No rows met exclusion criteria before {self.cfg.data.drop_patient_if_exclusion_before_date}. Didn't drop any.",
            )

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

        if not self.cfg.data.lookbehind_combination:
            raise ValueError("No lookbehind_combination provided.")

        # Extract all unique lookbhehinds in the dataset predictors
        lookbehinds_in_dataset = {
            int(infer_look_distance(col)[0])
            for col in infer_predictor_col_name(df=dataset)
            if len(infer_look_distance(col)) > 0
        }

        # Convert list to set
        lookbehinds_in_spec = set(self.cfg.data.lookbehind_combination)

        # Check that all loobehinds in lookbehind_combination are used in the predictors
        if not lookbehinds_in_spec.issubset(
            lookbehinds_in_dataset,
        ):
            msg.warn(
                f"One or more of the provided lookbehinds in lookbehind_combination is/are not used in any predictors in the dataset: {lookbehinds_in_spec - lookbehinds_in_dataset}",
            )

            lookbehinds_to_keep = lookbehinds_in_spec.intersection(
                lookbehinds_in_dataset,
            )

            if not lookbehinds_to_keep:
                raise ValueError("No predictors left after dropping lookbehinds.")

            msg.warn(f"Training on {lookbehinds_to_keep}.")
        else:
            lookbehinds_to_keep = lookbehinds_in_spec

        # Create a list of all predictor columns who have a lookbehind window not in lookbehind_combination list
        cols_to_drop = [
            c
            for c in infer_predictor_col_name(df=dataset)
            if all(str(l_beh) not in c for l_beh in lookbehinds_to_keep)
        ]

        cols_to_drop = [c for c in cols_to_drop if "within" in c]
        # ? Add some specification of within_x_days indicating how to parse columns to find lookbehinds. Or, alternatively, use the column spec.

        dataset = dataset.drop(columns=cols_to_drop)
        return dataset

    @staticmethod
    def convert_timestamp_dtype_and_nat(dataset: pd.DataFrame) -> pd.DataFrame:
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

        n_cols_before_modification = dataset.shape[1]

        if direction == "behind":
            cols_to_process = infer_predictor_col_name(df=dataset)

            for col in cols_to_process:
                # Extract lookbehind days from column name use regex
                # E.g. "column_name_within_90_days" == 90
                # E.g. "column_name_within_90_days_fallback_NaN" == 90
                lookbehind_days_strs = re.findall(r"within_(\d+)_days", col)

                if len(lookbehind_days_strs) > 0:
                    lookbehind_days = int(lookbehind_days_strs[0])
                else:
                    msg.warn(f"Could not extract lookbehind days from {col}")
                    continue

                if lookbehind_days > look_direction_threshold:
                    cols_to_drop.append(col)

        n_cols_after_modification = dataset.shape[1]
        percent_dropped = get_percent_lost(
            n_before=n_cols_before_modification,
            n_after=n_cols_after_modification,
        )

        if n_cols_before_modification - n_cols_after_modification != 0:
            msg.info(
                f"Dropped {n_cols_before_modification - n_cols_after_modification} ({percent_dropped}%) columns because they were looking {direction} further out than {look_direction_threshold} days.",
            )

        return dataset[[c for c in dataset.columns if c not in cols_to_drop]]

    def _drop_cols_and_rows_if_look_direction_not_met(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """Drop columns if they are outside the specification. Specifically:

        - min_lookahead_days is insufficient for the column's lookahead
        - The dataset doesn't stretch far enough for the prediction time's lookahead
        - The dataset doesn't stretch far enough for the prediction time's lookbehind

        Args:
            dataset (pd.DataFrame): Dataset to process.
        """
        for direction in ("ahead", "behind"):

            if direction in ("ahead", "behind"):
                if direction == "ahead":
                    n_days = self.cfg.data.min_lookahead_days
                elif direction == "behind":
                    n_days = max(self.cfg.data.lookbehind_combination)
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

    def _keep_unique_outcome_col_with_lookahead_days_matching_conf(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """Keep only one outcome column with the same lookahead days as set in
        the config."""
        outcome_cols = infer_outcome_col_name(df=dataset, allow_multiple=True)

        col_to_drop = [
            c for c in outcome_cols if str(self.cfg.data.min_lookahead_days) not in c
        ]

        # If no columns to drop, return the dataset
        if not col_to_drop:
            return dataset

        df = dataset.drop(col_to_drop, axis=1)

        if not len(infer_outcome_col_name(df)) == 1:
            raise ValueError(
                "Returning more than one outcome column, will cause problems during eval.",
            )

        return df

    def _keep_only_if_older_than_min_age(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows that are older than the minimum age specified in the
        config."""
        return dataset[dataset[self.cfg.data.col_name.age] >= self.cfg.data.min_age]

    def n_outcome_col_names(self, df: pd.DataFrame) -> int:
        """How many outcome columns there are in a dataframe."""
        return len(infer_outcome_col_name(df=df, allow_multiple=True))

    def _drop_rows_after_event_time(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Drop all rows where prediction timestamp is after the outcome."""

        rows_to_drop = (
            dataset[self.cfg.data.col_name.pred_timestamp]
            > dataset[self.cfg.data.col_name.outcome_timestamp]
        )

        return dataset[~rows_to_drop]

    def _convert_boolean_dtypes_to_int(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Convert boolean dtypes to int."""
        for col in dataset.columns:
            if dataset[col].dtype == bool:
                dataset[col] = dataset[col].astype(int)

        return dataset

    def _negative_values_to_nan(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Convert negative values to NaN."""
        preds = dataset[infer_predictor_col_name(df=dataset)]

        # Get all columns with negative values
        cols_with_numerical_values = preds.select_dtypes(include=["number"]).columns

        numerical_columns_with_negative_values = [
            c for c in cols_with_numerical_values if preds[c].min() < 0
        ]

        df_to_replace = dataset[numerical_columns_with_negative_values]

        # Convert to NaN
        df_to_replace[df_to_replace < 0] = np.nan
        dataset[numerical_columns_with_negative_values] = df_to_replace

        return dataset

    def _process_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Process dataset, namely:

        - Drop patients with outcome before drop_patient_if_outcome_before_date
        - Process timestamp columns
        - Drop visits where min_lookahead, min_lookbehind or min_prediction_time_date are not met
        - Drop features with lookbehinds not in lookbehind_combination

        Returns:
            pd.DataFrame: Processed dataset
        """
        # Super hacky rename, needs to be removed before merging. Figure out how to add eval columns when creating the dataset.
        dataset = dataset.rename(
            {
                "pred_hba1c_within_9999_days_count_fallback_nan": self.cfg.data.col_name.custom.n_hba1c,
            },
            axis=1,
        )

        dataset = self.convert_timestamp_dtype_and_nat(dataset=dataset)

        if self.cfg.preprocessing.convert_booleans_to_int:
            dataset = self._convert_boolean_dtypes_to_int(dataset=dataset)

        if self.cfg.data.min_age:
            dataset = self._keep_only_if_older_than_min_age(dataset=dataset)

        dataset = self._drop_rows_after_event_time(dataset=dataset)

        if self.cfg.data.drop_patient_if_exclusion_before_date:
            dataset = self._drop_patient_if_excluded(dataset=dataset)

        # Drop if later than min prediction time date
        if self.cfg.data.min_prediction_time_date:
            dataset = dataset[
                dataset[self.cfg.data.col_name.pred_timestamp]
                > self.cfg.data.min_prediction_time_date
            ]

        dataset = self._drop_cols_and_rows_if_look_direction_not_met(dataset=dataset)

        if self.cfg.data.lookbehind_combination:
            dataset = self._drop_cols_not_in_lookbehind_combination(dataset=dataset)

        dataset = self._keep_unique_outcome_col_with_lookahead_days_matching_conf(
            dataset=dataset,
        )

        return dataset

    @cache_pandas_result(cache_dir=FEATURE_SETS_PATH / "dataset_cache")
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
        msg.info(f"Loading {split_names}")

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


class SplitDataset(BaseModel):
    """A dataset split into train, test and optionally validation."""

    class Config:
        """Configuration for the dataclass to allow pd.DataFrame as type."""

        arbitrary_types_allowed = True

    train: pd.DataFrame
    test: Optional[pd.DataFrame] = None
    val: pd.DataFrame


def load_train_from_cfg(cfg: FullConfigSchema) -> pd.DataFrame:
    """Load train dataset from config.

    Args:
        cfg (FullConfig): Config

    Returns:
        pd.DataFrame: Train dataset
    """
    return DataLoader(cfg=cfg).load_dataset_from_dir(split_names="train")


def load_train_and_val_from_cfg(cfg: FullConfigSchema):
    """Load train and validation data from file."""

    loader = DataLoader(cfg=cfg)

    return SplitDataset(
        train=loader.load_dataset_from_dir(split_names="train"),
        val=loader.load_dataset_from_dir(split_names="val"),
    )


def get_latest_dataset_dir(path: Path) -> Path:
    """Get the latest dataset directory by time of creation."""
    return max(path.glob("*"), key=os.path.getctime)


def load_train_raw(cfg: FullConfigSchema):
    """Load the data."""
    path = Path(cfg.data.dir)
    file_names = list(path.glob(pattern=r"*train*"))

    if len(file_names) == 1:
        file_name = file_names[0]
        file_suffix = file_name.suffix
        if file_suffix == ".parquet":
            df = pd.read_parquet(file_name)
        elif file_suffix == ".csv":
            df = pd.read_csv(file_name)

        df = DataLoader.convert_timestamp_dtype_and_nat(dataset=df)

        return df

    raise ValueError(f"Returned {len(file_names)} files")
