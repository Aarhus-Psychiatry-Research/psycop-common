"""Testing of the utils module."""
# pylint: disable=missing-function-docstring

import numpy as np
import pandas as pd

from psycop_model_training.utils.utils import (
    drop_records_if_datediff_days_smaller_than,
    flatten_nested_dict,
)


def convert_cols_with_matching_colnames_to_datetime(
    df: pd.DataFrame,
    colname_substr: str,
) -> pd.DataFrame:
    """Convert columns that contain colname_substr in their name to datetimes.

    Args:
        df (pd.DataFrame): The dataframe to convert.
        colname_substr (str): Substring to match on.

    Returns:
        pd.DataFrame: The converted dataframe.
    """
    df.loc[:, df.columns.str.contains(colname_substr)] = df.loc[
        :,
        df.columns.str.contains(colname_substr),
    ].apply(pd.to_datetime)

    return df


def str_to_df(string, convert_timestamp_to_datetime: bool = True) -> pd.DataFrame:
    """Convert a string to a dataframe.

    Args:
        string (str): String to convert to a dataframe. String should be a csv.
        convert_timestamp_to_datetime (bool, optional): Whether to convert
            timestamp columns to datetime. Defaults to True.

    Returns:
        pd.DataFrame: The dataframe
    """
    from io import StringIO

    df = pd.read_table(StringIO(string), sep=",", index_col=False)

    if convert_timestamp_to_datetime:
        df = convert_cols_with_matching_colnames_to_datetime(df, "timestamp")

    # Drop "Unnamed" cols
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]


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
