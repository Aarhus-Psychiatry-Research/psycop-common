"“”Handle fast writing to SQL database.“”"

import urllib
import urllib.parse
from collections.abc import Sequence
from typing import Optional

from sqlalchemy import create_engine


def get_sql_table_names(
    driver: Optional[str] = "SQL Server",
    server: Optional[str] = "BI-DPA-PROD",
    database: Optional[str] = "USR_PS_FORSK",
) -> Sequence[str]:
    """Extracts SQL table names starting with 'psycop'.
    Args:
        driver (str, optional): The SQL driver. Defaults to "SQL Server".
        server (str, optional): The SQL server. Defaults to “BI-DPA_PROD”.
        database (str, optional): The SQL database. Defaults to “USR_PS_Forsk”.
    """

    # Driver for Kubeflow is different from driver on Ovartaci
    driver = "SQL Server"

    params = urllib.parse.quote(
        f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
    )

    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
    conn = engine.connect().execution_options(stream_results=True, fast_executemany=True)

    table_names = conn.execute(
        "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME LIKE 'psycop%';"
    ).fetchall()

    conn.close()  # type: ignore
    engine.dispose()  # type: ignore

    # return list of table names
    return [item[0] for item in table_names]
