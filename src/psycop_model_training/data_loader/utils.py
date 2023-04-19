import os
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from psycop_model_training.config_schemas.data import DataSchema
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.config_schemas.preprocessing import (
    PreSplitPreprocessingConfigSchema,
)
from psycop_model_training.data_loader.data_classes import SplitDataset
from psycop_model_training.data_loader.data_loader import DataLoader
from psycop_model_training.preprocessing.pre_split.full_processor import (
    pre_split_process_full_dataset,
)
from psycop_model_training.preprocessing.pre_split.processors.value_cleaner import (
    PreSplitValueCleaner,
)


def get_latest_dataset_dir(path: Path) -> Path:
    """Get the latest dataset directory by time of creation."""
    return max(path.glob("*"), key=os.path.getctime)


def load_and_filter_split_from_cfg(
    data_cfg: DataSchema,
    pre_split_cfg: PreSplitPreprocessingConfigSchema,
    split: Literal["train", "test", "val"],
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load train dataset from config.

    Args:
        data_cfg (DataSchema): Data config
        pre_split_cfg (PreSplitPreprocessingConfigSchema): Pre-split config
        split (Literal["train", "test", "val"]): Split to load
        cache_dir (Optional[Path], optional): Directory. Defaults to None, in which case no caching is used.

    Returns:
        pd.DataFrame: Train dataset
    """
    dataset = DataLoader(data_cfg=data_cfg).load_dataset_from_dir(split_names=split)
    filtered_data = pre_split_process_full_dataset(
        dataset=dataset,
        pre_split_cfg=pre_split_cfg,
        data_cfg=data_cfg,
        cache_dir=cache_dir,
    )

    return filtered_data


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


def load_and_filter_train_and_val_from_cfg(cfg: FullConfigSchema) -> SplitDataset:
    """Load train and validation data from file."""
    return SplitDataset(
        train=load_and_filter_split_from_cfg(
            pre_split_cfg=cfg.preprocessing.pre_split,
            data_cfg=cfg.data,
            split="train",
        ),
        val=load_and_filter_split_from_cfg(
            pre_split_cfg=cfg.preprocessing.pre_split,
            data_cfg=cfg.data,
            split="val",
        ),
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
