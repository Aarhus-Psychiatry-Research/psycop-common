from pathlib import Path

import pandas as pd

from psycop.common.model_training.config_schemas.data import ColumnNamesSchema, DataSchema
from psycop.common.model_training.data_loader.data_loader import DataLoader


def test_load_dataset_from_dir(
    tmpdir: str, base_feature_df: pd.DataFrame, feature_df_different_split: pd.DataFrame
):
    tmpdir_path = Path(tmpdir)
    train_data = base_feature_df
    train_data.to_parquet(tmpdir_path / "train.parquet")

    eval_data = feature_df_different_split
    eval_data.to_parquet(tmpdir_path / "eval.parquet")

    # Load datasets using DataLoader
    data_cfg = DataSchema(
        dir=tmpdir_path,
        splits_for_training=[""],
        n_training_samples=None,
        col_name=ColumnNamesSchema(outcome_timestamp=None, age=None, is_female=None),
    )

    data_loader = DataLoader(data_cfg)
    train_dataset = data_loader.load_dataset_from_dir(split_names="train", nrows=None)
    eval_dataset = data_loader.load_dataset_from_dir(split_names="eval", nrows=None)
    both_datasets = data_loader.load_dataset_from_dir(["train", "eval"], nrows=None)

    # Assert that datasets were loaded correctly
    assert train_dataset.equals(train_data)
    assert eval_dataset.equals(eval_data)
    assert both_datasets.equals(pd.concat([train_data, eval_data]).reset_index(drop=True))
