"""Dataset loader."""
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd
from wasabi import Printer

from psycop.common.model_training.config_schemas.data import DataSchema

msg = Printer(timestamp=True)

from psycop.common.model_training.data_loader.col_name_checker import (
    check_columns_exist_in_dataset,
)

log = logging.getLogger(__name__)


class DataLoader:
    """Class to handle loading of a datasplit."""

    def __init__(
        self,
        data_cfg: DataSchema,
        column_name_checker: Optional[Callable] = check_columns_exist_in_dataset,  # type: ignore
    ):
        self.data_cfg = data_cfg
        self.data_dir = data_cfg.dir
        # File handling
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

    @staticmethod
    def _check_dataframes_can_be_concatenated(
        datasets: list[pd.DataFrame],
        uuid_column: str,
    ) -> bool:
        """Check if pred_time_uuid columns are sorted so they can be concatenated
        instead of joined."""
        base_uuid = datasets[0][uuid_column]
        return all(base_uuid.equals(df[uuid_column]) for df in datasets[1:])

    @staticmethod
    def _check_dataframes_can_be_joined(
        datasets: list[pd.DataFrame],
        uuid_column: str,
    ) -> bool:
        """Check if pred_time_uuid columns contain the same elements so they can
        be joined without introducing new rows"""
        base_uuid = datasets[0][uuid_column]
        all_base_uuids_in_other_datasets = [
            base_uuid.isin(df[uuid_column]).all() for df in datasets[1:]
        ]
        return all(all_base_uuids_in_other_datasets)

    @staticmethod
    def _remove_id_columns(
        datasets: list[pd.DataFrame],
        id_columns: Sequence[str],
    ) -> list[pd.DataFrame]:
        """Remove id columns from all but the first dataset"""
        return [
            dataset.drop(columns=id_columns) if i > 0 else dataset
            for i, dataset in enumerate(datasets)
        ]

    def _check_and_merge_feature_sets(
        self,
        datasets: list[pd.DataFrame],
    ) -> pd.DataFrame:
        """Check if datasets can be concatenated or need to be joined."""
        n_rows_per_dataset = [dataset.shape[0] for dataset in datasets]
        all_datasets_have_same_length = len(set(n_rows_per_dataset)) == 1
        if not all_datasets_have_same_length:
            raise ValueError(
                """The datasets have a different amount of rows.
                Ensure that they have been created with the same prediction times.""",
            )
        shared_id_columns = [
            self.data_cfg.col_name.id,
            self.data_cfg.col_name.pred_time_uuid,
            self.data_cfg.col_name.pred_timestamp,
        ]
        if DataLoader._check_dataframes_can_be_concatenated(
            datasets=datasets,
            uuid_column=self.data_cfg.col_name.pred_time_uuid,
        ):
            log.debug("Concatenating multiple feature sets.")

            datasets = DataLoader._remove_id_columns(
                datasets=datasets,
                id_columns=shared_id_columns,
            )
            return pd.concat(datasets, axis=1)

        if DataLoader._check_dataframes_can_be_joined(
            datasets=datasets,
            uuid_column=self.data_cfg.col_name.pred_time_uuid,
        ):
            log.debug("Joining multiple feature sets.")
            merged_df = datasets[0]
            for df in datasets[1:]:
                merged_df = pd.merge(
                    merged_df,
                    df,
                    on=shared_id_columns,
                    how="outer",
                    validate="1:1",
                )
            return merged_df
        raise ValueError(
            "The datasets have different uuids. Ensure that they have been created with the same prediction times.",
        )

    def _load_dataset_file(
        self,
        split_name: str,
        dataset_dir: Path,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load dataset from directory. Finds any file with the matching file
        suffix with the split name in its filename.

        Args:
            split_name (str): Name of split, allowed are ["train", "test", "val"]
            dataset_dir (Path): Directory containing the dataset
            nrows (Optional[int]): Number of rows to load. Defaults to None, in which case
                all rows are loaded.
            self.file_suffix (str, optional): File suffix of the dataset. Defaults to "parquet".

        Returns:
            pd.DataFrame: The dataset
        """
        msg.info(f"Loading {split_name}")

        if self.file_suffix not in ("csv", "parquet"):
            raise ValueError(f"File suffix {self.file_suffix} not supported.")

        path = list(dataset_dir.glob(f"*{split_name}*.{self.file_suffix}"))[0]  # noqa

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
        if not isinstance(self.data_dir, list):
            dataset_dirs = [self.data_dir]
        else:
            dataset_dirs = self.data_dir

        feature_sets = [
            self._load_dataset_from_dir(
                split_names=split_names,
                nrows=nrows,
                dataset_dir=Path(dataset_dir),
            )
            for dataset_dir in dataset_dirs
        ]

        if len(feature_sets) == 1:
            return feature_sets[0]
        # else, concatenate/join them
        merged_datasets = self._check_and_merge_feature_sets(
            datasets=feature_sets,
        )
        return merged_datasets

    def _load_dataset_from_dir(
        self,
        split_names: Union[list[str], tuple[str], str],
        nrows: Optional[int],
        dataset_dir: Path,
    ) -> pd.DataFrame:
        match split_names:
            case str():
                # just return the single split
                return self._load_dataset_file(
                    split_name=split_names,
                    nrows=nrows,
                    dataset_dir=dataset_dir,
                )

            case list() | tuple():
                # Concat splits if multiple are given
                if nrows is not None:
                    nrows = int(
                        nrows / len(split_names),
                    )
                return pd.concat(
                    [
                        self._load_dataset_file(
                            split_name=split,
                            nrows=nrows,
                            dataset_dir=dataset_dir,
                        )
                        for split in split_names
                    ],
                    ignore_index=True,
                )
