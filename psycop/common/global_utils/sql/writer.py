"“”Handle fast writing to SQL database.“”"

import urllib
import urllib.parse
from collections.abc import Generator, Sequence
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection
from tqdm import tqdm
from wasabi import msg


def chunker(seq: Sequence | pd.DataFrame, size: int) -> Generator:  # type: ignore
    """Yield successive n-sized chunks from seq."""

    # from http://stackoverflow.com/a/434328
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def insert_with_progress(
    df: pd.DataFrame, table_name: str, conn: Connection, rows_per_chunk: int, if_exists: str
):
    """Chunk dataframe and insert each chunk, showing a progress bar.
    Args:
        df (pd.DataFrame): Dataframe to insert.
        table_name (str): SQL table name to insert into.
        conn (_type_): Connected sqlalchemy engine.
        rows_per_chunk (int): How many rows to fit into each chunk.
        if_exists (str): What to do if table exists. Takes {'fail', 'replace', 'append'}.s
    """

    with tqdm(total=len(df)) as pbar:
        for i, chunked_df in enumerate(  # pylint: disable=invalid-name
            chunker(df, rows_per_chunk)
        ):
            replace = if_exists if i == 0 else "append"
            chunked_df.to_sql(
                schema="fct", con=conn, name=table_name, if_exists=replace, index=False
            )
            pbar.update(rows_per_chunk)


def write_df_to_sql(
    df: pd.DataFrame,
    table_name: str,
    rows_per_chunk: int = 5000,
    server: Optional[str] = "BI-DPA-PROD",
    database: Optional[str] = "USR_PS_FORSK",
    if_exists: str = "fail",
):
    """Writes a pandas dataframe to the SQL server.
    Args:
        df (pd.DataFrame): dataframe to write
        table_name (str): name of table to write to
        rows_per_chunk (int): rows per chunk for loading bar
        server (str, optional): The SQL server. Defaults to “BI-DPA_PROD”.
        database (str, optional): The SQL database. Defaults to “USR_PS_Forsk”.
        if_exists (str): What to do if the table already exists. Takes {'fail', 'replace', 'append'}. Defaults to “fail”.
    """

    # Driver for Kubeflow is different from driver on Ovartaci
    driver = "SQL Server"

    params = urllib.parse.quote(
        f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
    )

    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
    conn = engine.connect().execution_options(stream_results=True, fast_executemany=True)
    if if_exists == "replace":
        msg.warn(
            "'replace' only replaces rows, not the table. If you want to delete rows, drop the entire table first (sql_load(query='DROP TABLE [fct].[psycop_train_ids]'))."
        )
    insert_with_progress(
        df=df,
        table_name=table_name,
        rows_per_chunk=rows_per_chunk,
        conn=conn,  # type: ignore
        if_exists=if_exists,
    )
    conn.close()  # type: ignore
    engine.dispose()  # type: ignore
