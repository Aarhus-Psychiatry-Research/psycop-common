"""Loaders for visits to psychiatry."""

import logging
from collections.abc import Sequence
from typing import Literal, Union

import pandas as pd
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_visits import RawValueSourceSchema
from psycop.common.feature_generation.loaders.raw.sql_load import sql_load

log = logging.getLogger(__name__)


def physical_visits_2024(
    timestamp_for_output: Literal["start", "end"] = "end",
    shak_code: Union[int, None] = None,
    shak_sql_operator: Union[str, None] = "=",
    where_clause: Union[str, None] = None,
    where_separator: Union[str, None] = "AND",
    n_rows: Union[int, None] = None,
    return_value_as_visit_length_days: Union[bool, None] = False,
    visit_types: Sequence[Literal["admissions", "ambulatory_visits", "emergency_visits"]] = (
        "admissions",
        "ambulatory_visits",
        "emergency_visits",
    ),
    return_shak_location: bool = False,
) -> pd.DataFrame:
    """Load pshysical visits to both somatic and psychiatry.

    Args:
        timestamp_for_output (Literal["start", "end"], optional): Whether to use the start or end timestamp for the output. Defaults to "end".
        shak_code (Optional[int], optional): SHAK code indicating where to keep/not keep visits from (e.g. 6600). Combines with
            shak_sql_operator, e.g. "!= 6600". Defaults to None, in which case all admissions are kept.
        shak_sql_operator (Optional[str], optional): Operator to use with shak_code. Defaults to "=".
        where_clause (Optional[str], optional): Extra where-clauses to add to the SQL call. E.g. dw_ek_borger = 1. Defaults to None. # noqa: DAR102
        where_separator (Optional[str], optional): Separator between where-clauses. Defaults to "AND".
        n_rows (Optional[int], optional): Number of rows to return. Defaults to None.
        return_value_as_visit_length_days (Optional[bool], optional): Whether to return length of visit in days as the value for the loader. Defaults to False which results in value=1 for all visits.
        visit_types (list[Literal["admissions", "ambulatory_visits", "emergency_visits"]]]): Which visit types to load. Defaults to ["admissions", "ambulatory_visits", "emergency_visits"].
        return_shak_location (bool): Whether to return the shak code of the visit

    Returns:
        pd.DataFrame: Dataframe with all physical visits to psychiatry. Has columns dw_ek_borger and timestamp.
    """

    # SHAK = 6600 ≈ in psychiatry

    source_schemas = {
        "LPR3": RawValueSourceSchema(
            view="[CVD_T2D_LPR3kontakter]",
            start_datetime_col_name="datotid_lpr3kontaktstart",
            end_datetime_col_name="datotid_lpr3kontaktslut",
            location_col_name="shakkode_lpr3kontaktansvarlig",
            where_clause="AND [Kontakttype] = 'Fysisk fremmøde'",
        ),
        "ambulatory_visits": RawValueSourceSchema(
            view="[CVD_T2D_besoeg_LPR2]",
            start_datetime_col_name="datotid_start",
            end_datetime_col_name="datotid_slut",
            location_col_name="shakafskode",
            where_clause="AND ambbesoeg = 1",
        ),
        "emergency_visits": RawValueSourceSchema(
            view="[CVD_T2D_akutambulantekontakter_LPR2]",
            start_datetime_col_name="datotid_start",
            end_datetime_col_name="datotid_slut",
            location_col_name="afsnit_stam",
            where_clause="",
        ),
        "admissions": RawValueSourceSchema(
            view="[CVD_T2D_indlaeggelser_LPR2]",
            start_datetime_col_name="datotid_indlaeggelse",
            end_datetime_col_name="datotid_udskrivning",
            location_col_name="shakKode_kontaktansvarlig",
            where_clause="",
        ),
    }

    allowed_visit_types = ["admissions", "ambulatory_visits", "emergency_visits"]

    if any(types not in allowed_visit_types for types in visit_types):
        raise ValueError(f"Invalid visit type. Allowed types of visits are {allowed_visit_types}.")

    english_to_lpr3_visit_type = {
        "admissions": "'Indlæggelse'",
        "ambulatory_visits": "'Ambulant'",
        "emergency_visits": "'Akut ambulant'",
    }
    chosen_schemas = {
        visit_type: source_schemas[visit_type] for visit_type in [*visit_types, "LPR3"]
    }
    english_to_lpr3_visit_type = [  # type: ignore
        english_to_lpr3_visit_type[visit] for visit in visit_types
    ]
    chosen_schemas[
        "LPR3"
    ].where_clause += f" AND pt_type IN ({','.join(english_to_lpr3_visit_type)})"

    dfs = []

    for schema in chosen_schemas.values():
        cols = f"{schema.start_datetime_col_name}, {schema.end_datetime_col_name}, dw_ek_borger, {schema.location_col_name}"

        sql = f"SELECT {cols} FROM [fct].{schema.view} WHERE {schema.start_datetime_col_name} IS NOT NULL {schema.where_clause}"

        if shak_code is not None:
            sql += f" AND {schema.location_col_name} != 'Ukendt'"
            sql += f" AND left({schema.location_col_name}, {len(str(shak_code))}) {shak_sql_operator} {shak_code!s}"

        if where_clause is not None:
            sql += f" {where_separator} {where_clause}"

        df = sql_load(sql, database="USR_PS_FORSK", n_rows=n_rows)
        df = df.rename(
            columns={
                schema.end_datetime_col_name: "timestamp_end",
                schema.start_datetime_col_name: "timestamp_start",
                schema.location_col_name: "shak_location",
            }
        )

        dfs.append(df)

    # Concat the list of dfs
    output_df = pd.concat(dfs)

    # Round timestamps to whole seconds before dropping duplicates
    output_timestamp_col_name = f"timestamp_{timestamp_for_output}"

    output_df[output_timestamp_col_name] = output_df[output_timestamp_col_name].dt.round("1s")
    output_df = output_df.drop_duplicates(
        subset=[output_timestamp_col_name, "dw_ek_borger"], keep="first"
    )
    output_df = output_df.dropna(subset=[output_timestamp_col_name])  # type: ignore

    # Change value column to length of admission in days
    if return_value_as_visit_length_days:
        output_df["value"] = (
            output_df["timestamp_end"] - pd.to_datetime(output_df["timestamp_start"])
        ).dt.total_seconds() / 86400
    else:
        output_df["value"] = 1

    log.info("Loaded physical visits")

    output_df = output_df.rename(columns={output_timestamp_col_name: "timestamp"})
    output_cols = ["dw_ek_borger", "timestamp", "value"]
    if return_shak_location:
        output_cols = [*output_cols, "shak_location"]

    return output_df[output_cols].reset_index(drop=True)


def physical_visits_loader_2024(
    n_rows: Union[int, None] = None, return_value_as_visit_length_days: Union[bool, None] = False
) -> pd.DataFrame:
    """Load physical visits to all units."""
    return physical_visits_2024(
        n_rows=n_rows, return_value_as_visit_length_days=return_value_as_visit_length_days
    )


def physical_visits_to_psychiatry_2024(
    n_rows: Union[int, None] = None,
    timestamps_only: bool = False,
    return_value_as_visit_length_days: Union[bool, None] = True,
    timestamp_for_output: Literal["start", "end"] = "start",
) -> pd.DataFrame:
    """Load physical visits to psychiatry."""
    df = physical_visits_2024(
        shak_code=6600,
        shak_sql_operator="=",
        n_rows=n_rows,
        return_value_as_visit_length_days=return_value_as_visit_length_days,
        timestamp_for_output=timestamp_for_output,
    )

    if timestamps_only:
        df = df.drop(columns=["value"])

    return df


def physical_visits_to_somatic_2024(
    n_rows: Union[int, None] = None, return_value_as_visit_length_days: Union[bool, None] = False
) -> pd.DataFrame:
    """Load physical visits to somatic."""
    return physical_visits_2024(
        shak_code=6600,
        shak_sql_operator="!=",
        n_rows=n_rows,
        return_value_as_visit_length_days=return_value_as_visit_length_days,
    )


def admissions_2024(
    n_rows: Union[int, None] = None,
    return_value_as_visit_length_days: Union[bool, None] = False,
    shak_code: Union[int, None] = None,
    shak_sql_operator: Union[str, None] = None,
) -> pd.DataFrame:
    """Load admissions."""
    return physical_visits_2024(
        visit_types=["admissions"],
        return_value_as_visit_length_days=return_value_as_visit_length_days,
        n_rows=n_rows,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
    )


def ambulatory_visits_2024(
    n_rows: Union[int, None] = None,
    return_value_as_visit_length_days: Union[bool, None] = False,
    shak_code: Union[int, None] = None,
    shak_sql_operator: Union[str, None] = None,
    timestamps_only: bool = False,
    timestamp_for_output: Literal["start", "end"] = "end",
) -> pd.DataFrame:
    """Load ambulatory visits."""
    df = physical_visits_2024(
        timestamp_for_output=timestamp_for_output,
        visit_types=["ambulatory_visits"],
        return_value_as_visit_length_days=return_value_as_visit_length_days,
        n_rows=n_rows,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
    )
    if timestamps_only:
        df = df.drop(columns=["value"])

    return df


def emergency_visits_2024(
    n_rows: Union[int, None] = None,
    return_value_as_visit_length_days: Union[bool, None] = False,
    shak_code: Union[int, None] = None,
    shak_sql_operator: Union[str, None] = None,
) -> pd.DataFrame:
    """Load emergency visits."""
    return physical_visits_2024(
        visit_types=["emergency_visits"],
        return_value_as_visit_length_days=return_value_as_visit_length_days,
        n_rows=n_rows,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
    )


def first_visit_to_psychiatry_2024() -> pl.DataFrame:
    first_visit = (
        pl.from_pandas(
            physical_visits_to_psychiatry_2024(
                n_rows=None,
                timestamps_only=False,
                return_value_as_visit_length_days=False,
                timestamp_for_output="start",
            )
        )
        .groupby("dw_ek_borger")
        .agg(pl.col("timestamp").min())
    )
    return first_visit
