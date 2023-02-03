"""Loaders for visits to psychiatry."""

import logging
from typing import Optional, Literal

import pandas as pd

from psycop_feature_generation.loaders.raw.sql_load import sql_load
from psycop_feature_generation.utils import data_loaders

log = logging.getLogger(__name__)


@data_loaders.register("physical_visits")
def physical_visits(
    shak_code: Optional[int] = None,
    shak_sql_operator: Optional[str] = "=",
    where_clause: Optional[str] = None,
    where_separator: Optional[str] = "AND",
    n_rows: Optional[int] = None,
    visit_length: Optional[bool] = False,
    visit_type: Optional[
        Literal["admissions", "ambulatory_visits", "emergency_visits"]
    ] = None,
) -> pd.DataFrame:
    """Load pshysical visits to both somatic and psychiatry.

    Args:
        shak_code (Optional[int], optional): SHAK code indicating where to keep/not keep visits from (e.g. 6600). Combines with
            shak_sql_operator, e.g. "!= 6600". Defaults to None, in which case all admissions are kept.
        shak_sql_operator (Optional[str], optional): Operator to use with shak_code. Defaults to "=".
        where_clause (Optional[str], optional): Extra where-clauses to add to the SQL call. E.g. dw_ek_borger = 1. Defaults to None. # noqa: DAR102
        where_separator (Optional[str], optional): Separator between where-clauses. Defaults to "AND".
        n_rows (Optional[int], optional): Number of rows to return. Defaults to None.
        visit_length...
        visit_type...

    Returns:
        pd.DataFrame: Dataframe with all physical visits to psychiatry. Has columns dw_ek_borger and timestamp.
    """

    # SHAK = 6600 ≈ in psychiatry
    d = {
        "LPR3": {
            "view": "[FOR_LPR3kontakter_psyk_somatik_inkl_2021_feb2022]",
            "datetime_col": "datotid_lpr3kontaktslut",
            "value_col": "datotid_lpr3kontaktstart",
            "location_col": "shakkode_lpr3kontaktansvarlig",
            "where": "AND [Kontakttype] = 'Fysisk fremmøde'",
        },
        "ambulatory_visits": {
            "view": "[FOR_besoeg_psyk_somatik_LPR2_inkl_2021_feb2022]",
            "datetime_col": "datotid_slut",
            "value_col": "datotid_start",
            "location_col": "shakafskode",
            "where": "AND ambbesoeg = 1",
        },
        "emergency_visits": {
            "view": "[FOR_akutambulantekontakter_psyk_somatik_LPR2_inkl_2021_feb2022]",
            "datetime_col": "datotid_slut",
            "value_col": "datotid_start",
            "location_col": "afsnit_stam",
            "where": "",
        },
        "admissions": {
            "view": "[FOR_indlaeggelser_psyk_somatik_LPR2_inkl_2021_feb2022]",
            "datetime_col": "datotid_udskrivning",
            "value_col": "datotid_indlaeggelse",
            "location_col": "shakKode_kontaktansvarlig",
            "where": "",
        },
    }

    if visit_type:
        LPR3_types = {
            "admissions": "'Indlæggelse'",
            "ambulatory_visits": "'Ambulant'",
            "emergency_visits": "'Akut ambulant'",
        }
        d = {key: d[key] for key in ["LPR3", visit_type]}
        d["LPR3"]["where"] += f" AND pt_type = {LPR3_types[visit_type]}"

    dfs = []

    for meta in d.values():
        cols = f"{meta['datetime_col']}, dw_ek_borger"

        if visit_length:
            cols += f", {meta['value_col']}"

        sql = f"SELECT {cols} FROM [fct].{meta['view']} WHERE {meta['datetime_col']} IS NOT NULL {meta['where']}"

        if shak_code is not None:
            sql += f" AND left({meta['location_col']}, {len(str(shak_code))}) {shak_sql_operator} {str(shak_code)}"

        if where_clause is not None:
            sql += f" {where_separator} {where_clause}"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n_rows=n_rows)
        df.rename(columns={meta["datetime_col"]: "timestamp"}, inplace=True)

        dfs.append(df)

    # Concat the list of dfs
    output_df = pd.concat(dfs)

    # 0,8% of visits are duplicates. Unsure if overlap between sources or errors in source data. Removing.
    output_df = output_df.drop_duplicates(
        subset=["timestamp", "dw_ek_borger"],
        keep="first",
    )

    if visit_length:
        output_df.rename(columns={"value_col": "value"})
    else:
        output_df["value"] = 1

    log.info("Loaded physical visits")

    return output_df.reset_index(drop=True)


@data_loaders.register("physical_visits")
def physical_visits(n_rows: Optional[int] = None) -> pd.DataFrame:
    """Load physical visits to all units."""
    return physical_visits(n_rows=n_rows)


@data_loaders.register("physical_visits_to_psychiatry")
def physical_visits_to_psychiatry(
    n_rows: Optional[int] = None,
    timestamps_only: bool = True,
) -> pd.DataFrame:
    """Load physical visits to psychiatry."""
    df = physical_visits(shak_code=6600, shak_sql_operator="=", n_rows=n_rows)

    if timestamps_only:
        df = df.drop("value", axis=1)

    return df


@data_loaders.register("physical_visits_to_somatic")
def physical_visits_to_somatic(n_rows: Optional[int] = None) -> pd.DataFrame:
    """Load physical visits to somatic."""
    return physical_visits(shak_code=6600, shak_sql_operator="!=", n_rows=n_rows)


@data_loaders.register("admissions")
def admissions(
    n_rows: Optional[int] = None,
    visit_length: Optional[bool] = False,
) -> pd.DataFrame:
    """Load physical visits to somatic."""
    return physical_visits(
        visit_type="admissions", visit_length=visit_length, n_rows=n_rows
    )
