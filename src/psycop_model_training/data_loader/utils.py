import os
from pathlib import Path

import pandas as pd

from psycop_model_training.config.schemas import FullConfigSchema
from psycop_model_training.data_loader.data_classes import SplitDataset
from psycop_model_training.data_loader.data_loader import DataLoader


def get_latest_dataset_dir(path: Path) -> Path:
    """Get the latest dataset directory by time of creation."""
    return max(path.glob("*"), key=os.path.getctime)


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
