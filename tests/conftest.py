"""Define fixtures for tests."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from psycop_model_training.config_schemas.conf_utils import (
    FullConfigSchema,
    load_test_cfg_as_pydantic,
)
from psycop_model_training.model_eval.dataclasses import EvalDataset

CONFIG_DIR_PATH_REL = "../application/config"


def add_age_and_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Add age and gender columns to dataframe.

    Args:
        df (pd.DataFrame): The dataframe to add age and sex cols to.
    """
    ids = pd.DataFrame({"dw_ek_borger": df["dw_ek_borger"].unique()})
    ids["age"] = np.random.randint(17, 95, len(ids))
    ids["sex"] = np.where(ids["dw_ek_borger"] > 30_000, "F", "M")

    return df.merge(ids)


def add_eval_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add eval_ column to dataframe to test table 1 functionality.

    Args:
        df (pd.DataFrame): The dataframe to add age
    """
    df["eval_n_hbac1_count"] = np.random.randint(0, 20, len(df))

    return df

@pytest.fixture(scope="function")
def synth_eval_dataset() -> EvalDataset:
    """Load synthetic data."""
    csv_path = Path("tests") / "test_data" / "synth_eval_data.csv"
    df = pd.read_csv(csv_path)
    df = add_age_and_sex(df)
    df = add_eval_column(df)

    # Convert all timestamp cols to datetime
    for col in [col for col in df.columns if "timestamp" in col]:
        df[col] = pd.to_datetime(df[col])

    return EvalDataset(
        ids=df["dw_ek_borger"],
        y=df["label"],
        y_hat_probs=df["pred_prob"],
        y_hat_int=df["pred"],
        pred_timestamps=df["timestamp"],
        outcome_timestamps=df["timestamp_t2d_diag"],
        age=df["age"],
        sex=df["sex"],
        custom_columns={"eval_n_hbac1_count": df["eval_n_hbac1_count"]},
    )


@pytest.fixture(scope="function")
def immuteable_test_config() -> FullConfigSchema:
    """Get an immutable config for testing."""
    return load_test_cfg_as_pydantic(
        config_file_name="default_config.yaml",
        allow_mutation=False,
    )


@pytest.fixture(scope="function")
def muteable_test_config() -> FullConfigSchema:
    """Get a mutable config for testing."""
    return load_test_cfg_as_pydantic(
        config_file_name="default_config.yaml",
        allow_mutation=True,
    )
