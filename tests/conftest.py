"""Define fixtures for tests."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from psycop_model_training.config_schemas.conf_utils import (
    FullConfigSchema,
    load_test_cfg_as_pydantic,
)
from psycop_model_training.training_output.dataclasses import EvalDataset

CONFIG_DIR_PATH_REL = "../application/config"


def add_eval_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add eval_ column to dataframe to test table 1 functionality.

    Args:
        df (pd.DataFrame): The dataframe to add age
    """
    df["eval_n_hbac1_count"] = np.random.randint(0, 20, len(df))

    return df


@pytest.fixture()
def synth_eval_dataset() -> EvalDataset:
    """Load synthetic data."""
    csv_path = Path("tests") / "test_data" / "model_eval" / "synth_eval_data.csv"
    df = pd.read_csv(csv_path)

    # Convert all timestamp cols to datetime
    for col in [col for col in df.columns if "timestamp" in col]:
        df[col] = pd.to_datetime(df[col])

    df = add_eval_column(df)

    return EvalDataset(
        ids=df["dw_ek_borger"],
        y=df["label"],
        y_hat_probs=df["pred_prob"],
        pred_timestamps=df["timestamp"],
        outcome_timestamps=df["timestamp_t2d_diag"],
        age=df["age"],
        is_female=df["is_female"],
        custom_columns={"eval_n_hbac1_count": df["eval_n_hbac1_count"]},
    )


@pytest.fixture()
def immuteable_test_config() -> FullConfigSchema:
    """Get an immutable config for testing."""
    return load_test_cfg_as_pydantic(
        config_file_name="default_config.yaml",
    )


@pytest.fixture()
def muteable_test_config() -> FullConfigSchema:
    """Get a mutable config for testing."""
    return load_test_cfg_as_pydantic(
        config_file_name="default_config.yaml",
        allow_mutation=True,
    )
