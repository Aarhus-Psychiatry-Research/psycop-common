"""
Script for obtaining and writing all somatic emergency contacts
for all admissions from 2012-2022. Handles LPR2 to LPR3 transition, duplicates
and short term readmissions. Bemærk that Traumecenter er ambulant så kontakter udelukkende hertil tæller ikke
hvilket er helt ok da patienter der udskrives direkte fra Traumecenteret ikke har været syge (for så var de jo blevet indlagt).
Jeg skal dog være opmærksom på patienter som indlægges pga selvmordsforsøg. Dem vil der være nogen af.
"""
import pandas as pd

from psycop.common.global_utils.sql.loader import sql_load
from psycop.common.global_utils.sql.writer import write_df_to_sql
from psycop.projects.forced_admission_inpatient.cohort.extract_admissions_and_visits.utils.utils import (
    concat_readmissions_for_all_patients,
    lpr2_lpr3_overlap,
)


def get_contacts_to_somatic_emergency(write: bool = False) -> pd.DataFrame:
    # Load contact data
    view = "[FOR_kohorte_indhold_pt_journal_psyk_somatik_inkl_2021_feb2022]"
    cols_to_keep = "datotid_start, datotid_slut, dw_ek_borger, pt_type"

    sql = "SELECT " + cols_to_keep + " FROM [fct]." + view

    #sql = "SELECT * FROM [fct]." + view
    sql += "WHERE datotid_start > '2012-01-01' AND pt_type = 'Indlagt'"
    sql += " AND datotid_start IS NOT NULL AND datotid_slut IS NOT NULL;"

    df = pd.DataFrame(sql_load(sql, chunksize=None))  # type: ignore

    df["datotid_start"] = df["datotid_start"].apply(
        pd.to_datetime
    )

    df["datotid_slut"] = df["datotid_slut"].apply(
        pd.to_datetime
    )


    if write:
        ROWS_PER_CHUNK = 5_000

        write_df_to_sql(
            df=df[["dw_ek_borger", "datotid_start"]],
            table_name="all_psychiatric_outpatient_visits_processed_2012_2021_ANDDAN_SOMATIC_ADMISSION",
            if_exists="replace",
            rows_per_chunk=ROWS_PER_CHUNK,
        )

    #Så sætter jeg det korrekte navn - de andre kalder nedenstående funktion der gør det samme. Måske for at spare tid når datasættet loades
    df = df.rename(columns={"datotid_start": "timestamp"})

    return df[["dw_ek_borger", "timestamp"]]  # type: ignore

def admissions_onset_timestamps() -> pd.DataFrame:
    # Load somatic_admissions data
    view = "[all_psychiatric_outpatient_visits_processed_2012_2021_ANDDAN_SOMATIC_ADMISSION]"
    cols_to_keep = "dw_ek_borger, datotid_start"

    sql = "SELECT " + cols_to_keep + " FROM [fct]." + view

    admissions_onset_timestamps = pd.DataFrame(sql_load(sql, chunksize=None))  # type: ignore

    admissions_onset_timestamps = admissions_onset_timestamps.rename(
        columns={"datotid_start": "timestamp"}
    )

    return admissions_onset_timestamps

if __name__ == "__main__":
    get_contacts_to_somatic_emergency()