"""Loaders for demographic information."""

from __future__ import annotations

import pandas as pd

from psycop_feature_generation.loaders.raw.sql_load import sql_load
from psycop_feature_generation.utils import data_loaders


@data_loaders.register("birthdays")
def birthdays(n_rows: int | None = None) -> pd.DataFrame:
    view = "[FOR_kohorte_demografi_inkl_2021_feb2022]"

    sql = f"SELECT dw_ek_borger, foedselsdato FROM [fct].{view}"

    df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n_rows=n_rows)

    # Typically handled by sql_load, but because foedselsdato doesn't contain "datotid" in its name,
    # We must handle it manually here
    df["foedselsdato"] = pd.to_datetime(df["foedselsdato"], format="%Y-%m-%d")

    df = df.rename(columns={"foedselsdato": "date_of_birth"})

    return df.reset_index(drop=True)


@data_loaders.register("sex_female")
def sex_female(n_rows: int | None = None) -> pd.DataFrame:
    view = "[FOR_kohorte_demografi_inkl_2021_feb2022]"

    sql = f"SELECT dw_ek_borger, koennavn FROM [fct].{view}"

    df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n_rows=n_rows)

    df.loc[df["koennavn"] == "Mand", "koennavn"] = False
    df.loc[df["koennavn"] == "Kvinde", "koennavn"] = True

    df = df.rename(
        columns={"koennavn": "sex_female"},
    )

    return df.reset_index(drop=True)
