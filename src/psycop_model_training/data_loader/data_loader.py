"""Loader for the t2d dataset."""
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from wasabi import Printer

from psycop_model_training.utils.config_schemas import FullConfigSchema

msg = Printer(timestamp=True)


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
        msg.info(f"Loading {split_name}")

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

    def _process_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Process dataset, namely:

        - Drop patients with outcome before drop_patient_if_outcome_before_date
        - Process timestamp columns
        - Drop visits where min_lookahead, min_lookbehind or min_prediction_time_date are not met
        - Drop features with lookbehinds not in lookbehind_combination

        Returns:
            pd.DataFrame: Processed dataset
        """
        msg = Printer(timestamp=True)
        msg.info("Processing dataset")

        # Super hacky rename, needs to be removed before merging. Figure out how to add eval columns when creating the dataset.
        dataset = dataset.rename(
            {
                "pred_hba1c_within_9999_days_count_fallback_nan": self.cfg.data.col_name.custom.n_hba1c,
            },
            axis=1,
        )

        # Super hacky transformation of negative weights (?!) for chi-square.
        # In the future, we want to:
        # 1. Fix this in the feature generation for t2d
        # 2a. See if there's a way of using feature selection that permits negative values, or
        # 2b. Always use z-score normalisation?
        dataset = self._negative_values_to_nan(dataset=dataset)

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

        msg.info("Finished processing dataset")

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
