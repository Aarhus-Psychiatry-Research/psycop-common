import pandas as pd


def str_to_df(str, convert_timestamp_to_datetime: bool = True) -> pd.DataFrame:
    from io import StringIO

    df = pd.read_table(StringIO(str), sep=",", index_col=False)

    if convert_timestamp_to_datetime:
        df = convert_cols_with_matching_colnames_to_datetime(df, "timestamp")

    # Drop "Unnamed" cols
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]


def convert_cols_with_matching_colnames_to_datetime(
    df: pd.DataFrame,
    colname_substr: str,
) -> pd.DataFrame:
    """Convert columns that contain colname_substr in their name to datetimes
    Args:
        df (pd.DataFrame)
        colname_substr (str): Substring to match on.
    """
    df.loc[:, df.columns.str.contains(colname_substr)] = df.loc[
        :,
        df.columns.str.contains(colname_substr),
    ].apply(pd.to_datetime)

    return df
