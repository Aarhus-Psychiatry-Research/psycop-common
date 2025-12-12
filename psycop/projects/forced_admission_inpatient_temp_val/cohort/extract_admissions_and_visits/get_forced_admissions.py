"""Script for mapping the forced admission data from the SEI data to
the more finegrained MidtEPJ admission data
"""

import pandas as pd

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load


def get_forced_admissions_2025() -> pd.DataFrame:
    # Load coercion data
    view = "[FOR_tvang_tvangsindlæggelser_hele_kohorten_tvangsindlæggelse_sep_2025]"
    cols_to_keep = "datotid_start_sei, datotid_slut_sei, dw_ek_borger, typetekst_sei"

    sql = "SELECT " + cols_to_keep + " FROM [fct]." + view

    forced_admissions = pd.DataFrame(sql_load(sql)).drop_duplicates()  # type: ignore
    forced_admissions[["datotid_start_sei", "datotid_slut_sei"]] = forced_admissions[  # type: ignore
        ["datotid_start_sei", "datotid_slut_sei"]
    ].apply(pd.to_datetime)

    return forced_admissions  # type: ignore


def forced_admissions_onset_timestamps_2025(
    timestamps_only: bool = False, timestamp_as_value_col: bool = False
) -> pd.DataFrame:
    # Load forced_admissions data
    view = "[FOR_tvang_tvangsindlæggelser_hele_kohorten_tvangsindlæggelse_sep_2025]"
    cols_to_keep = "dw_ek_borger, datotid_start_sei"

    sql = "SELECT " + cols_to_keep + " FROM [fct]." + view

    forced_admissions = pd.DataFrame(sql_load(sql, n_rows=None)).drop_duplicates()  # type: ignore

    forced_admissions = forced_admissions.rename(  # type: ignore
        columns={"datotid_start_sei": "timestamp"}
    )

    if timestamp_as_value_col:
        forced_admissions["value"] = forced_admissions["timestamp"].copy()
    else:
        forced_admissions["value"] = 1

    if timestamps_only:
        return forced_admissions[["dw_ek_borger", "timestamp"]]

    return forced_admissions


def forced_admissions_end_timestamps_2025() -> pd.DataFrame:
    # Load forced_admissions data
    view = "[FOR_tvang_tvangsindlæggelser_hele_kohorten_tvangsindlæggelse_sep_2025]"
    cols_to_keep = "dw_ek_borger, datotid_slut_sei"

    sql = "SELECT " + cols_to_keep + " FROM [fct]." + view

    forced_admissions = pd.DataFrame(sql_load(sql, n_rows=None)).drop_duplicates()  # type: ignore

    forced_admissions = forced_admissions.rename(  # type: ignore
        columns={"datotid_slut_sei": "timestamp"}
    )

    return forced_admissions


if __name__ == "__main__":
    get_forced_admissions_2025()
    forced_admissions_end_timestamps_2025()
