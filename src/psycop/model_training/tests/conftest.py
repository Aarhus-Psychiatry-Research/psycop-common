"""Define fixtures for tests."""

import numpy as np
import pandas as pd
import pytest
from psycop.model_training.config_schemas.conf_utils import (
    FullConfigSchema,
    load_test_cfg_as_pydantic,
)
from psycop.model_training.training_output.dataclasses import EvalDataset
from psycop.utils import PSYCOP_PKG_ROOT

CONFIG_DIR_PATH_REL = "../application/config"


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests",
    )


def pytest_configure(config: pytest.Config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def add_eval_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add eval_ column to dataframe to test table 1 functionality.

    Args:
        df (pd.DataFrame): The dataframe to add age
    """
    df["eval_n_hbac1_count"] = np.random.randint(0, 20, len(df))

    return df


@pytest.fixture()
def synth_eval_df() -> pd.DataFrame:
    """Load synthetic data."""
    csv_path = PSYCOP_PKG_ROOT / "test_utils" / "model_eval" / "synth_eval_data.csv"
    df = pd.read_csv(csv_path)

    # Convert all timestamp cols to datetime
    for col in [col for col in df.columns if "timestamp" in col]:
        df[col] = pd.to_datetime(df[col])

    return df


@pytest.fixture()
def synth_eval_dataset(synth_eval_df: pd.DataFrame) -> EvalDataset:
    """Load synthetic data."""
    df = synth_eval_df

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
        pred_time_uuids=df["dw_ek_borger"].astype(str)
        + df["timestamp"].dt.strftime(
            "-%Y-%m-%d-%H-%M-%S",
        ),
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
