import pandas as pd


def convert_cols_with_matching_colnames_to_datetime(
    df: pd.DataFrame, colname_substr: str
) -> pd.DataFrame:
    """Convert columns that contain colname_substr in their name to datetimes.

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
