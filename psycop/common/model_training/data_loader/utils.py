import os
from pathlib import Path
from typing import Optional

import pandas as pd

from psycop.common.model_training.config_schemas.data import DataSchema
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.common.model_training.config_schemas.preprocessing import (
    PreSplitPreprocessingConfigSchema,
)
from psycop.common.model_training.data_loader.data_loader import DataLoader
from psycop.common.model_training.preprocessing.pre_split.full_processor import (
    pre_split_process_full_dataset,
)
from psycop.common.model_training.preprocessing.pre_split.processors.value_cleaner import (
    PreSplitValueCleaner,
)


def check_dataframes_can_be_concatenated(
    datasets: list[pd.DataFrame],
    uuid_column: str,
) -> bool:
    """Check if pred_time_uuid columns are sorted so they can be concatenated
    instead of joined."""
    base_uuid = datasets[0][uuid_column]
    return all(base_uuid.equals(df[uuid_column]) for df in datasets[1:])


def check_and_merge_feature_sets(
    datasets: list[pd.DataFrame],
    uuid_column: str,
) -> pd.DataFrame:
    n_rows = [dataset.shape[0] for dataset in datasets]
    if len(set(n_rows)) != 1:
        raise ValueError(
            "The datasets have a different amount of rows. "
            + "Ensure that they have been created with the same "
            + "prediction times.",
        )

    if check_dataframes_can_be_concatenated(datasets=datasets, uuid_column=uuid_column):
        return pd.concat(datasets, axis=1)
    merged_df = datasets[0]
    for df in datasets[1:]:
        merged_df = pd.merge(merged_df, df, on=uuid_column, how="outer", validate="1:1")
    return merged_df


def get_latest_dataset_dir(path: Path) -> Path:
    """Get the latest dataset directory by time of creation."""
    return max(path.glob("*"), key=os.path.getctime)


def load_and_filter_split_from_cfg(
    data_cfg: DataSchema,
    pre_split_cfg: PreSplitPreprocessingConfigSchema,
    split: str,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load train dataset from config. If data_cfg.dir is a list of paths to
    different feature sets, will concatenate them

    Args:
        data_cfg (DataSchema): Data config
        pre_split_cfg (PreSplitPreprocessingConfigSchema): Pre-split config
        split (str): Split to load
        cache_dir (Optional[Path], optional): Directory. Defaults to None, in which case no caching is used.

    Returns:
        pd.DataFrame: Train dataset
    """

    if not isinstance(data_cfg.dir, list):
        dataset_dirs = [data_cfg.dir]
    else:
        dataset_dirs = data_cfg.dir

    datasets = [
        DataLoader(data_cfg=data_cfg).load_dataset_from_dir(
            split_names=split,
            dataset_dir=dataset_dir,
        )
        for dataset_dir in dataset_dirs
    ]

    filtered_data = [
        pre_split_process_full_dataset(
            dataset=dataset,
            pre_split_cfg=pre_split_cfg,
            data_cfg=data_cfg,
            cache_dir=cache_dir,
        )
        for dataset in datasets
    ]

    if len(filtered_data) == 1:
        return filtered_data[0]

    merged_datasets = check_and_merge_feature_sets(
        datasets=filtered_data,
        uuid_column=data_cfg.col_name.pred_time_uuid,
    )
    return merged_datasets


def load_and_filter_train_from_cfg(
    cfg: FullConfigSchema,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load train dataset from config.

    Args:
        cfg (FullConfig): Config
        cache_dir (Optional[Path], optional): Directory. Defaults to None, in which case no caching is used.

    Returns:
        pd.DataFrame: Train dataset
    """
    return load_and_filter_split_from_cfg(
        pre_split_cfg=cfg.preprocessing.pre_split,
        data_cfg=cfg.data,
        split="train",
        cache_dir=cache_dir,
    )


def load_train_raw(
    cfg: FullConfigSchema,
    convert_timestamp_types_and_nans: bool = True,
) -> pd.DataFrame:
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
        else:
            raise ValueError(f"File suffix {file_suffix} not supported")

        # Helpful during tests to convert columns with matching names to datetime
        if convert_timestamp_types_and_nans:
            return PreSplitValueCleaner.convert_timestamp_dtype_and_nat(dataset=df)

    raise ValueError(f"Returned {len(file_names)} files")
