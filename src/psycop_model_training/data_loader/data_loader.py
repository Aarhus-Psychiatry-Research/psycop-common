"""Dataset loader."""
import logging
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd
from psycop_model_training.config_schemas.data import DataSchema
from wasabi import Printer

msg = Printer(timestamp=True)

from psycop_model_training.data_loader.col_name_checker import (
    check_columns_exist_in_dataset,
)

log = logging.getLogger(__name__)


class DataLoader:
    """Class to handle loading of a datasplit."""

    def __init__(
        self,
        data_cfg: DataSchema,
        column_name_checker: Optional[Callable] = check_columns_exist_in_dataset,
    ):
        self.data_cfg = data_cfg

        # File handling
        self.dir_path = Path(data_cfg.dir)
        self.file_suffix = data_cfg.suffix
        self.column_name_checker = column_name_checker

        # Column specifications
        self.pred_col_name_prefix = data_cfg.pred_prefix

    def _check_column_names(self, df: pd.DataFrame):
        """Check that all columns in the config exist in the dataset."""
        if self.column_name_checker:
            self.column_name_checker(col_name_schema=self.data_cfg.col_name, df=df)
        else:
            log.debug("No column name checker specified. Skipping column name check.")

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

            df: pd.DataFrame = pd.read_parquet(path)
        elif "csv" in self.file_suffix:
            df: pd.DataFrame = pd.read_csv(filepath_or_buffer=path, nrows=nrows)
        else:
            raise ValueError(f"File suffix {self.file_suffix} not supported.")

        if self.column_name_checker:
            self._check_column_names(df=df)

        return df

    def load_dataset_from_dir(
        self,
        split_names: Union[list[str], tuple[str], str],
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load dataset. Can load multiple splits at once, e.g. concatenate
        train and val for crossvalidation.

        Args:
            split_names (Union[Sequence[str], str]): Name of split, allowed are ["train", "test", "val"]
            nrows (Optional[int]): Number of rows to load from dataset. Defaults to None, in which case all rows are loaded.

        Returns:
            pd.DataFrame: The filtered dataset
        """
        # Concat splits if multiple are given
        if isinstance(split_names, (list, tuple)):
            if isinstance(split_names, list):
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

        # Otherwise, just return the single split
        return self._load_dataset_file(split_name=split_names, nrows=nrows)
