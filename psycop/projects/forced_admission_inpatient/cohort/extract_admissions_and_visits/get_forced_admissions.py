"""Script for mapping the forced admission data from the SEI data to
the more finegrained MidtEPJ admission data
"""

import pandas as pd

from psycop.common.global_utils.sql.loader import sql_load
from psycop.common.global_utils.sql.writer import write_df_to_sql


def get_forced_admissions(write: bool = True) -> pd.DataFrame:
    # Load coercion data
    view = "[FOR_tvang_alt_hele_kohorten_inkl_2021_feb2022]"
    cols_to_keep = "datotid_start_sei, datotid_slut_sei, dw_ek_borger, typetekst_sei"

    sql = "SELECT " + cols_to_keep + " FROM [fct]." + view
    sql += (
        "WHERE datotid_start_sei > '2012-01-01' AND typetekst_sei = 'Tvangsindlæggelse'"
    )

    forced_admissions = pd.DataFrame(sql_load(sql, chunksize=None)).drop_duplicates()  # type: ignore
    forced_admissions[["datotid_start_sei", "datotid_slut_sei"]] = forced_admissions[
        ["datotid_start_sei", "datotid_slut_sei"]
    ].apply(pd.to_datetime)

    if write:
        ROWS_PER_CHUNK = 5_000

        write_df_to_sql(
            df=forced_admissions,
            table_name="forced_admissions_processed_2012_2021",
            if_exists="replace",
            rows_per_chunk=ROWS_PER_CHUNK,
        )
    return forced_admissions


def forced_admissions_onset_timestamps(timestamps_only: bool = False) -> pd.DataFrame:
    # Load forced_admissions data
    view = "[forced_admissions_processed_2012_2021]"
    cols_to_keep = "dw_ek_borger, datotid_start_sei"

    sql = "SELECT " + cols_to_keep + " FROM [fct]." + view

    forced_admissions = pd.DataFrame(sql_load(sql, chunksize=None)).drop_duplicates()  # type: ignore

    forced_admissions = forced_admissions.rename(
        columns={"datotid_start_sei": "timestamp"},
    )

    forced_admissions["value"] = 1

    if timestamps_only:
        return forced_admissions[["dw_ek_borger", "timestamp"]]

    return forced_admissions


def forced_admissions_end_timestamps() -> pd.DataFrame:
    # Load forced_admissions data
    view = "[forced_admissions_processed_2012_2021]"
    cols_to_keep = "dw_ek_borger, datotid_slut_sei"

    sql = "SELECT " + cols_to_keep + " FROM [fct]." + view

    forced_admissions = pd.DataFrame(sql_load(sql, chunksize=None)).drop_duplicates()  # type: ignore

    forced_admissions = forced_admissions.rename(
        columns={"datotid_slut_sei": "timestamp"},
    )

    return forced_admissions


if __name__ == "__main__":
    get_forced_admissions(write=True)
    forced_admissions_end_timestamps()
