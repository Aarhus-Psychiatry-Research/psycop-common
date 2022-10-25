"""Testing of the utils module."""
# pylint: disable=missing-function-docstring
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize
from utils_for_testing import str_to_df

from psycopt2d.utils.configs import omegaconf_to_pydantic_objects
from psycopt2d.utils.utils import (
    PROJECT_ROOT,
    drop_records_if_datediff_days_smaller_than,
    flatten_nested_dict,
)


def test_drop_records_if_datediff_days_smaller_than():
    test_df = str_to_df(
        """timestamp_2,timestamp_1
    2021-01-01, 2021-01-31,
    2021-01-30, 2021-01-01,
    2021-01-01, 2021-01-01,
    """,
    )

    test_df = test_df.append(
        pd.DataFrame({"timestamp_2": pd.NaT, "timestamp_1": "2021-01-01"}, index=[1]),
    )

    drop_records_if_datediff_days_smaller_than(
        df=test_df,
        t2_col_name="timestamp_2",
        t1_col_name="timestamp_1",
        threshold_days=1,
        inplace=True,
    )

    differences = (
        (test_df["timestamp_2"] - test_df["timestamp_1"]) / np.timedelta64(1, "D")
    ).to_list()
    expected = [29.0, np.nan]

    for i, diff in enumerate(differences):
        if np.isnan(diff):
            assert np.isnan(expected[i])
        else:
            assert diff == expected[i]


def test_flatten_nested_dict():
    input_dict = {"level1": {"level2": 3}}
    expected_dict = {"level1.level2": 3}

    output_dict = flatten_nested_dict(input_dict)

    assert expected_dict == output_dict


CONFIG_DIR_PATH_ABS = PROJECT_ROOT / "src" / "psycopt2d" / "config"
CONFIG_DIR_PATH_REL = "../src/psycopt2d/config"


def get_config_file_names() -> list[str]:
    """Get all config file names."""
    config_file_paths: list[Path] = list(CONFIG_DIR_PATH_ABS.glob("*.yaml"))
    return [f"{path.stem}.yaml" for path in config_file_paths]


@pytest.mark.parametrize("config_file_name", get_config_file_names())
def test_configs(config_file_name):
    with initialize(version_base=None, config_path=CONFIG_DIR_PATH_REL):
        cfg = compose(
            config_name=config_file_name,
        )

    cfg = omegaconf_to_pydantic_objects(conf=cfg)
