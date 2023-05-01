from __future__ import annotations

import pandas as pd


def keep_rows_where_col_name_matches_pattern(
    df: pd.DataFrame,
    col_name: str,
    regex_pattern: str,
):
    return df[df[col_name].str.contains(regex_pattern)]


def keep_rows_where_diag_matches_t2d_diag(
    df: pd.DataFrame,
    col_name: str,
) -> pd.DataFrame:
    """
    Keep rows where the diagnosis matches the T2D diagnosis.
    """
    t2d_regex_pattern = r"(:DE1[1-5].*)|(:DE16[0-2].*)|(:DO24.*)|(:DT383A.*)|(:DM142.*)|(:DG590.*)|(:DG632*)|(:DH280.*)|(:DH334.*)|(:DH360.*)|(:DH450.*)|(:DN083.*)"
    df = keep_rows_where_col_name_matches_pattern(
        df=df,
        col_name=col_name,
        regex_pattern=t2d_regex_pattern,
    )

    return df


def keep_rows_where_diag_matches_t1d_diag(
    df: pd.DataFrame,
    col_name: str,
) -> pd.DataFrame:
    """
    Keep rows where the diagnosis matches the T1D diagnosis.
    """
    t2d_regex_pattern = r"(:DE10.*)|(:DO240.*)"
    df = keep_rows_where_col_name_matches_pattern(
        df=df,
        col_name=col_name,
        regex_pattern=t2d_regex_pattern,
    )

    return df
