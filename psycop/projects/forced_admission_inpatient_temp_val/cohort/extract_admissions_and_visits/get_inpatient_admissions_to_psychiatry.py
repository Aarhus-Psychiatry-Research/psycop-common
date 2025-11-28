"""
Script for obtaining and writing all admission  and discharge timestamps
for all admissions from 2012-2022. Handles LPR2 to LPR3 transition, duplicates
and short term readmissions
"""

import pandas as pd

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.projects.forced_admission_inpatient_temp_val.cohort.extract_admissions_and_visits.utils.utils import (
    concat_readmissions_for_all_patients,
)


def get_admissions_to_psychiatry_2025() -> pd.DataFrame:
    # Load contact data
    view = "[FOR_kohorte_indhold_pt_journal_tvangsindlæggelse_sep_2025]"
    cols_to_keep = "datotid_start, datotid_slut, dw_ek_borger, pt_type"

    sql = "SELECT " + cols_to_keep + " FROM [fct]." + view
    sql += "WHERE pt_type = 'Indlagt'"
    sql += "AND datotid_start IS NOT NULL AND datotid_slut IS NOT NULL;"

    df = pd.DataFrame(sql_load(sql))  # type: ignore

    df[["datotid_start", "datotid_slut"]] = df[["datotid_start", "datotid_slut"]].apply(
        pd.to_datetime
    )

    # Aggregate readmissions within 4 hours to one admission
    df = concat_readmissions_for_all_patients(df)

    return df[["dw_ek_borger", "datotid_start", "datotid_slut"]]


def admissions_onset_timestamps_2025() -> pd.DataFrame:
    # Load forced_admissions data
    view = "[FOR_kohorte_indhold_pt_journal_tvangsindlæggelse_sep_2025]"
    cols_to_keep = "datotid_start, dw_ek_borger"

    sql = "SELECT " + cols_to_keep + " FROM [fct]." + view

    df = pd.DataFrame(sql_load(sql))  # type: ignore

    df[["datotid_start", "datotid_slut"]] = df[["datotid_start", "datotid_slut"]].apply(
        pd.to_datetime
    )

    # Aggregate readmissions within 4 hours to one admission
    df = concat_readmissions_for_all_patients(df)

    admissions_onset_timestamps = df.rename(columns={"datotid_star": "timestamp"})

    return admissions_onset_timestamps


def admissions_discharge_timestamps_2025() -> pd.DataFrame:
    # Load forced_admissions data
    view = "[FOR_kohorte_indhold_pt_journal_tvangsindlæggelse_sep_2025]"
    cols_to_keep = "datotid_slut, dw_ek_borger"

    sql = "SELECT " + cols_to_keep + " FROM [fct]." + view

    df = pd.DataFrame(sql_load(sql))  # type: ignore

    df[["datotid_start", "datotid_slut"]] = df[["datotid_start", "datotid_slut"]].apply(
        pd.to_datetime
    )

    # Aggregate readmissions within 4 hours to one admission
    df = concat_readmissions_for_all_patients(df)

    admissions_discharge_timestamps = df.rename(columns={"datotid_slut": "timestamp"})

    return admissions_discharge_timestamps


if __name__ == "__main__":
    get_admissions_to_psychiatry_2025()
