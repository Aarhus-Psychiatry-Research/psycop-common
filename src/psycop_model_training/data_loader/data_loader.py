"""Dataset loader."""
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd
from psycop_model_training.config_schemas.full_config import FullConfigSchema
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
        cfg: FullConfigSchema,
        column_name_checker: Optional[Callable] = check_columns_exist_in_dataset,
    ):
        self.cfg: FullConfigSchema = cfg

        # File handling
        self.dir_path = Path(cfg.data.dir)
        self.file_suffix = cfg.data.suffix
        self.column_name_checker = column_name_checker

        # Column specifications
        self.pred_col_name_prefix = cfg.data.pred_prefix

    def _check_column_names(self, df: pd.DataFrame):
        """Check that all columns in the config exist in the dataset."""
        if self.column_name_checker:
            self.column_name_checker(col_name_schema=self.cfg.data.col_name, df=df)
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

            df = pd.read_parquet(path)
        elif "csv" in self.file_suffix:
            df = pd.read_csv(filepath_or_buffer=path, nrows=nrows)

        if self.column_name_checker:
            self._check_column_names(df=df)

        return df

    def load_dataset_from_dir(
        self,
        split_names: Union[Iterable[str], str],
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load dataset. Can load multiple splits at once, e.g. concatenate
        train and val for crossvalidation.

        Args:
            split_names (Union[Iterable[str], str]): Name of split, allowed are ["train", "test", "val"]
            nrows (Optional[int]): Number of rows to load from dataset. Defaults to None, in which case all rows are loaded.

        Returns:
            pd.DataFrame: The filtered dataset
        """
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
        return dataset
