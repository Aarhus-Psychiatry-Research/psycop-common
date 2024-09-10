"""Example of."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


import logging
import urllib
import urllib.parse

import pandas as pd
from sqlalchemy import create_engine, text

from psycop.automation.environment import on_ovartaci

log = logging.getLogger(__name__)


def sql_load(
    query: str,
    server: str = "BI-DPA-PROD",
    database: str = "USR_PS_Forsk",
    format_timestamp_cols_to_datetime: bool | None = True,
    n_rows: int | None = None,
) -> pd.DataFrame:
    """Function to load a SQL query. If chunksize is None, all data will be
    loaded into memory. Otherwise, will stream the data in chunks of chunksize
    as a generator.

    Args:
        query (str): The SQL query
        server (str): The BI server
        database (str): The BI database
        format_timestamp_cols_to_datetime (bool, optional): Whether to format all
            columns with "datotid" in their name as pandas datetime. Defaults to true.
        n_rows (int, optional): Defaults to None. If specified, only returns the first n rows.

    Returns:
        Union[pd.DataFrame, Generator[pd.DataFrame]]: DataFrame or generator of DataFrames

    Example:
        # From USR_PS_Forsk
        >>> view = "[FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret_2011]"
        >>> sql = "SELECT * FROM [fct]." + view
        >>> df = sql_load(sql, chunksize = None)
    """
    # Driver for Kubeflow is different from driver on Ovartaci
    driver = "SQL Server" if on_ovartaci() else "ODBC Driver 18 for SQL Server"
    params = urllib.parse.quote(
        f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes"
    )

    if n_rows:
        query = query.replace("SELECT", f"SELECT TOP {n_rows} ")

    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

    conn = engine.connect().execution_options(stream_results=True, fast_executemany=True)
    log.info(f"Loading {query}")
    df = pd.read_sql(text(query), conn)  # type: ignore

    if format_timestamp_cols_to_datetime:
        datetime_col_names = [
            colname
            for colname in df.columns
            if any(substr in colname.lower() for substr in ["datotid", "timestamp"])
        ]

        df[datetime_col_names] = df[datetime_col_names].apply(pd.to_datetime)

    conn.close()  # type: ignore
    engine.dispose()  # type: ignore

    return df


def str_to_sql_match_logic(
    code_to_match: str, code_sql_col_name: str, load_diagnoses: bool, match_with_wildcard: bool
) -> str:
    """Generate SQL match logic from a single string.

    Args:
        code_to_match (list[str]): List of strings to match.
        code_sql_col_name (str): Name of the SQL column containing the codes.
        load_diagnoses (bool): Whether to load diagnoses or medications. Determines the logic. See calling function for more.
        match_with_wildcard (bool): Whether to match on icd_code* / atc_code* or only icd_code / atc_code.
    """
    if load_diagnoses:
        base_query = f"lower({code_sql_col_name}) LIKE '%{code_to_match.lower()}"
    else:
        base_query = f"lower({code_sql_col_name}) LIKE '{code_to_match.lower()}"

    if match_with_wildcard:
        return f"{base_query}%'"

    if load_diagnoses:
        return f"{base_query}' OR {base_query}#%'"

    return f"{base_query}'"


def list_to_sql_logic(
    codes_to_match: list[str],
    code_sql_col_name: str,
    load_diagnoses: bool,
    match_with_wildcard: bool,
) -> str:
    """Generate SQL match logic from a list of strings.

    Args:
        codes_to_match (list[str]): List of strings to match.
        code_sql_col_name (str): Name of the SQL column containing the codes.
        load_diagnoses (bool): Whether to load diagnoses or medications. Determines the logic. See calling function for more.
        match_with_wildcard (bool): Whether to match on icd_code* / atc_code* or only icd_code / atc_code.
    """
    match_col_sql_strings = []

    for code_str in codes_to_match:
        if load_diagnoses:
            base_query = f"lower({code_sql_col_name}) LIKE '%{code_str.lower()}"
        else:
            base_query = f"lower({code_sql_col_name}) LIKE '{code_str.lower()}"

        if match_with_wildcard:
            match_col_sql_strings.append(f"{base_query}%'")
        else:
            # If the string is at the end of diagnosegruppestreng, it doesn't end with a hashtag
            match_col_sql_strings.append(f"{base_query}'")

            if load_diagnoses:
                # If the string is at the beginning of diagnosegruppestreng, it doesn't start with a hashtag
                match_col_sql_strings.append(
                    f"lower({code_sql_col_name}) LIKE '{code_str.lower()}#%'"
                )

    return " OR ".join(match_col_sql_strings)
