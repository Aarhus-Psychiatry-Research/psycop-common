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


#[FOR_kohorte_indhold_pt_journal_psyk_somatik_inkl_2021_feb2022]

def get_contacts_to_somatic_emergency(write: bool = False) -> pd.DataFrame:
    # Load contact data
    view = "[FOR_indlaeggelser_psyk_somatik_LPR2_inkl_2021_feb2022]"
    cols_to_keep = "datotid_indlaeggelse, datotid_udskrivning, dw_ek_borger, pattypetekst, akutindlaeggelse, shakKode_kontaktansvarlig"

    sql = "SELECT " + cols_to_keep + " FROM [fct]." + view
    #sql = "SELECT * FROM [fct]." + view

    sql += "WHERE datotid_indlaeggelse > '2012-01-01'"
    sql += " AND pattypetekst = 'Indlagt' AND akutindlaeggelse = 'true' AND SUBSTRING(shakKode_kontaktansvarlig, 1, 4) != '6600'"
    sql += " AND datotid_indlaeggelse IS NOT NULL AND datotid_udskrivning IS NOT NULL;"

    df = pd.DataFrame(sql_load(sql, chunksize=None))  # type: ignore

    df["datotid_indlaeggelse"] = df["datotid_indlaeggelse"].apply(
        pd.to_datetime
    )

    df["datotid_udskrivning"] = df["datotid_udskrivning"].apply(
        pd.to_datetime
    )

    #Så sætter jeg det korrekte navn - de andre kalder nedenstående funktion der gør det samme. Måske for at spare tid når datasættet loades
    df = df.rename(columns={"datotid_indlaeggelse": "timestamp"})

    return df[["dw_ek_borger", "timestamp"]]  # type: ignore

if __name__ == "__main__":
    get_contacts_to_somatic_emergency()