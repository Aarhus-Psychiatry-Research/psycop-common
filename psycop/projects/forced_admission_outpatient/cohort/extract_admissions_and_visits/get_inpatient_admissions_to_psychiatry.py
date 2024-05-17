"""
Script for obtaining and writing all admission  and discharge timestamps
for all admissions from 2012-2022. Handles LPR2 to LPR3 transition, duplicates
and short term readmissions
"""
import pandas as pd

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.common.global_utils.sql.writer import write_df_to_sql
from psycop.projects.forced_admission_outpatient.cohort.extract_admissions_and_visits.utils.utils import (
    concat_readmissions_for_all_patients,
    lpr2_lpr3_overlap,
)


def get_admissions_to_psychiatry(write: bool = True) -> pd.DataFrame:
    # Load contact data
    view = "[FOR_kohorte_indhold_pt_journal_inkl_2021_feb2022]"
    cols_to_keep = "datotid_start, datotid_slut, dw_ek_borger, pt_type"

    sql = "SELECT " + cols_to_keep + " FROM [fct]." + view
    sql += "WHERE datotid_start > '2012-01-01' AND pt_type = 'Indlagt'"
    sql += "AND datotid_start IS NOT NULL AND datotid_slut IS NOT NULL;"

    df = pd.DataFrame(sql_load(sql, chunksize=None))  # type: ignore

    df[["datotid_start", "datotid_slut"]] = df[["datotid_start", "datotid_slut"]].apply(
        pd.to_datetime
    )

    # Resolve lpr2/lpr3 transition
    df = lpr2_lpr3_overlap(df)

    # Aggregate readmissions within 4 hours to one admission
    df = concat_readmissions_for_all_patients(df)

    if write:
        ROWS_PER_CHUNK = 5_000

        write_df_to_sql(
            df=df[["dw_ek_borger", "datotid_start", "datotid_slut"]],
            table_name="all_admissions_processed_2012_2021",
            if_exists="replace",
            rows_per_chunk=ROWS_PER_CHUNK,
        )

    return df[["dw_ek_borger", "datotid_start", "datotid_slut"]]


def admissions_onset_timestamps() -> pd.DataFrame:
    # Load forced_admissions data
    view = "[all_admissions_processed_2012_2021]"
    cols_to_keep = "datotid_start, dw_ek_borger"

    sql = "SELECT " + cols_to_keep + " FROM [fct]." + view

    admissions_onset_timestamps = pd.DataFrame(sql_load(sql, chunksize=None))  # type: ignore

    admissions_onset_timestamps = admissions_onset_timestamps.rename(
        columns={"datotid_star": "timestamp"}
    )

    return admissions_onset_timestamps


def admissions_discharge_timestamps() -> pd.DataFrame:
    # Load forced_admissions data
    view = "[all_admissions_processed_2012_2021]"
    cols_to_keep = "datotid_slut, dw_ek_borger"

    sql = "SELECT " + cols_to_keep + " FROM [fct]." + view

    admissions_discharge_timestamps = pd.DataFrame(sql_load(sql, chunksize=None))  # type: ignore

    admissions_discharge_timestamps = admissions_discharge_timestamps.rename(
        columns={"datotid_slut": "timestamp"}
    )

    return admissions_discharge_timestamps


if __name__ == "__main__":
    get_admissions_to_psychiatry()
