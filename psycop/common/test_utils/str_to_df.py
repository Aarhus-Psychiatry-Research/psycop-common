from io import StringIO

import numpy as np
import pandas as pd
import polars as pl
from pandas import DataFrame


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
    ].apply(
        pd.to_datetime,
    )
    return df


def str_to_df(
    string: str,
    convert_timestamp_to_datetime: bool = True,
    convert_np_nan_to_nan: bool = True,
    convert_str_to_float: bool = False,
    add_pred_time_uuid: bool = False,
    entity_id_colname: str = "entity_id",
    timestamp_col_name: str = "timestamp",
) -> DataFrame:
    """Convert a string representation of a dataframe to a dataframe.
    Args:
        string (str): A string representation of a dataframe.
        convert_timestamp_to_datetime (bool): Whether to convert the timestamp column to datetime. Defaults to True.
        convert_np_nan_to_nan (bool): Whether to convert np.nan to np.nan. Defaults to True.
        convert_str_to_float (bool): Whether to convert strings to floats. Defaults to False.
        add_pred_time_uuid (bool): Whether to infer a pred_time_uuid column from entity_id and timestamp columns. Defaults to False.
        entity_id_colname (str): The name of the entity_id column. Defaults to "entity_id".
        timestamp_col_name (str): The name of the timestamp column. Defaults to "timestamp".
    Returns:
        DataFrame: A dataframe.
    """
    # Drop comments for each line if any exist inside the str
    lines = []
    for i, line in enumerate(string.split("\n")):
        is_header = i == 0
        if is_header:
            # Remove all whitespace
            line = "".join(line.split())  # noqa: PLW2901
        # Remove leading whitespace
        if " #" in line:
            line = line[: line.rfind("#")]  # noqa

        line_sans_leading_trailing_whitespace = line.strip()
        line_without_ending_comma = (
            line_sans_leading_trailing_whitespace[:-1]
            if line_sans_leading_trailing_whitespace.endswith(",")
            else line
        )
        lines.append(line_without_ending_comma)

    full_string = "\n".join(list(lines))

    df = pd.read_table(StringIO(full_string), sep=",", index_col=False)

    if convert_timestamp_to_datetime:
        df = convert_cols_with_matching_colnames_to_datetime(df, "timestamp")

    if convert_np_nan_to_nan:
        # Convert "np.nan" str to the actual np.nan
        df = df.replace("np.nan", np.nan)

    if convert_str_to_float:
        # Convert all str to float
        df = df.apply(pd.to_numeric, axis=0, errors="coerce")

    if add_pred_time_uuid:
        df["pred_time_uuid"] = (
            df[entity_id_colname].astype(str) + "_" + df[timestamp_col_name].astype(str)
        )

    # Drop "Unnamed" cols
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]


def str_to_pl_df(
    string: str,
    convert_timestamp_to_datetime: bool = True,
    convert_np_nan_to_nan: bool = True,
    convert_str_to_float: bool = False,
    add_pred_time_uuid: bool = False,
    entity_id_colname: str = "entity_id",
    timestamp_col_name: str = "timestamp",
) -> pl.DataFrame:
    pd_df = str_to_df(
        string=string,
        convert_timestamp_to_datetime=convert_timestamp_to_datetime,
        convert_np_nan_to_nan=convert_np_nan_to_nan,
        convert_str_to_float=convert_str_to_float,
        add_pred_time_uuid=add_pred_time_uuid,
        entity_id_colname=entity_id_colname,
        timestamp_col_name=timestamp_col_name,
    )

    pl_df = pl.from_pandas(pd_df)

    return pl_df
