"""Utility functions for SQL loading."""

from __future__ import annotations

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
    database: str = "USR_PS_FORSK",
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
    driver = "SQL Server"

    # Separate setup for kubeflow
    if not on_ovartaci():
        driver = "ODBC Driver 18 for SQL Server"
        server = "rmsqls0175.onerm.dk"

    params = urllib.parse.quote(
        f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
    )

    if not on_ovartaci():
        params += "TrustServerCertificate=yes;"

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
