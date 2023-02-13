import os
from pathlib import Path
from typing import Literal

import pandas as pd

from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.data_loader.data_classes import SplitDataset
from psycop_model_training.data_loader.data_loader import DataLoader
from psycop_model_training.preprocessing.pre_split.full_processor import FullProcessor
from psycop_model_training.preprocessing.pre_split.processors.value_cleaner import (
    PreSplitValueCleaner,
)


def get_latest_dataset_dir(path: Path) -> Path:
    """Get the latest dataset directory by time of creation."""
    return max(path.glob("*"), key=os.path.getctime)


def load_and_filter_split_from_cfg(
    cfg: FullConfigSchema,
    split: Literal["train", "test", "val"],
) -> pd.DataFrame:
    """Load train dataset from config.

    Args:
        cfg (FullConfig): Config
        split (Literal["train", "test", "val"]): Split to load

    Returns:
        pd.DataFrame: Train dataset
    """
    dataset = DataLoader(cfg=cfg).load_dataset_from_dir(split_names=split)
    filtered_data = FullProcessor(cfg=cfg).process(dataset=dataset)

    return filtered_data


def load_and_filter_train_from_cfg(cfg: FullConfigSchema) -> pd.DataFrame:
    """Load train dataset from config.

    Args:
        cfg (FullConfig): Config

    Returns:
        pd.DataFrame: Train dataset
    """
    return load_and_filter_split_from_cfg(cfg=cfg, split="train")


def load_and_filter_train_and_val_from_cfg(cfg: FullConfigSchema):
    """Load train and validation data from file."""
    return SplitDataset(
        train=load_and_filter_split_from_cfg(cfg=cfg, split="train"),
        val=load_and_filter_split_from_cfg(cfg=cfg, split="val"),
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

        # Helpful during tests to convert columns with matching names to datetime
        if convert_timestamp_types_and_nans:
            df = PreSplitValueCleaner.convert_timestamp_dtype_and_nat(dataset=df)

        return df

    raise ValueError(f"Returned {len(file_names)} files")
