"""Utilites for testing."""

from typing import Any

import pandas as pd
import pytest
from pandas import DataFrame

from psycop.common.feature_generation.loaders.synth.raw.load_synth_data import (
    load_synth_outcome,
    load_synth_prediction_times,
)
from psycop.common.feature_generation.utils import data_loaders
from psycop.common.test_utils.str_to_df import str_to_df


def convert_cols_with_matching_colnames_to_datetime(
    df: DataFrame, colname_substr: str
) -> DataFrame:
    """Convert columns that contain colname_substr in their name to datetimes
    Args:
        df (DataFrame): The df to convert. # noqa: DAR101
        colname_substr (str): Substring to match on. # noqa: DAR101

    Returns:
        DataFrame: The converted df
    """
    df.loc[:, df.columns.str.contains(colname_substr)] = df.loc[
        :, df.columns.str.contains(colname_substr)
    ].apply(pd.to_datetime)

    return df


@data_loaders.register("load_event_times")
def load_event_times() -> pd.DataFrame:
    """Load event times."""
    event_times_str = """dw_ek_borger,timestamp,value,
                    1,2021-12-30 00:00:01, 1
                    1,2021-12-29 00:00:02, 2
                    """

    return str_to_df(event_times_str)


def check_any_item_in_list_has_str(list_of_str: list[Any], str_: str) -> bool:
    """Check if any item in a list contains a string.

    Args:
        list_of_str (list): A list of strings.
        str_ (str): A string.

    Returns:
        bool: True if any item in the list contains the string.
    """
    return any(str_ in item for item in list_of_str)


@pytest.fixture
def synth_prediction_times() -> pd.DataFrame:
    """Load the prediction times."""
    return load_synth_prediction_times()


@pytest.fixture
def synth_outcome() -> pd.DataFrame:
    """Load the synth outcome times."""
    return load_synth_outcome()
