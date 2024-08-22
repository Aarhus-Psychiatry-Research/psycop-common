"""
Script for obtaining and writing all somatic emergency contacts (defineret som akut somatiske indlæggelser)
for all admissions from 2012-2022. Handles LPR2 to LPR3 transition, duplicates
and short term readmissions. Bemærk that Traumecenter er ambulant så kontakter udelukkende hertil tæller ikke
hvilket er helt ok da patienter der udskrives direkte fra Traumecenteret ikke har været syge (for så var de jo blevet indlagt). Derudover, inkludere jeg kun akutte indlæggelser.
Jeg skal dog være opmærksom på patienter som indlægges pga selvmordsforsøg og dobbeltindlæggelser. Dem vil der være nogen af. 
"""
import pandas as pd

from psycop.common.global_utils.sql.loader import sql_load

def get_contacts_to_somatic_emergency(timestamps_only: bool = False, timestamp_as_value_col: bool = False) -> pd.DataFrame:
 #først hentes LPR2 data
    cols_to_keep = "datotid_indlaeggelse, datotid_udskrivning, dw_ek_borger, pattypetekst, akutindlaeggelse, shakKode_kontaktansvarlig, adiagnosekode"
    view = "[FOR_indlaeggelser_psyk_somatik_LPR2_inkl_2021_feb2022] "


    sql = "SELECT " + cols_to_keep + " FROM [fct]." + view
    #sql = "SELECT * FROM [fct]." + view

    sql += "WHERE datotid_indlaeggelse > '2012-01-01'"
    sql += " AND pattypetekst = 'Indlagt' AND akutindlaeggelse = 'true' AND SUBSTRING(shakKode_kontaktansvarlig, 1, 4) != '6600'"
    sql += " AND datotid_indlaeggelse IS NOT NULL AND datotid_udskrivning IS NOT NULL;"
    #sql += " AND SUBSTRING(adiagnosekode,2,2) IN ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N');"

    df_LPR2 = pd.DataFrame(sql_load(sql, chunksize=None)).drop_duplicates()  # type: ignore

    #Da jeg ikke kan finde ud af at sortere på diagnoser i SQL-kaldet bliver jeg nød til at gøre det nu (for udvælgelse af diagnoser se filen onedrive/forskning/søren østergaard/PSYCOP/akut somatisk indlæggelse/Akutte somatiske indlæggelsesdiagnoser adiagnoser)
    df_LPR2 = df_LPR2[df_LPR2['adiagnosekode'].str.slice(1, 2).isin(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N'])]

#så henter vi LPR3 data
    cols_to_keep = "datotid_lpr3kontaktstart, datotid_lpr3kontaktslut, dw_ek_borger, pt_type, prioritet, shakkode_lpr3kontaktansvarlig, adiagnosekode"
    view = "[FOR_LPR3kontakter_psyk_somatik_inkl_2021_feb2022] "

    sql = "SELECT " + cols_to_keep + " FROM [fct]." + view
    #sql = "SELECT * FROM [fct]." + view

    sql += "WHERE datotid_lpr3kontaktstart > '2012-01-01'"
    sql += " AND pt_type = 'Indlæggelse' AND prioritet = 'Akut' AND SUBSTRING(shakkode_lpr3kontaktansvarlig, 1, 4) != '6600'"
    sql += " AND datotid_lpr3kontaktstart IS NOT NULL AND datotid_lpr3kontaktslut IS NOT NULL;"

    df_LPR3 = pd.DataFrame(sql_load(sql, chunksize=None)).drop_duplicates()  # type: ignore

    #Da jeg ikke kan finde ud af at sortere på diagnoser i SQL-kaldet bliver jeg nød til at gøre det nu (for udvælgelse af diagnoser se filen onedrive/forskning/søren østergaard/PSYCOP/akut somatisk indlæggelse/Akutte somatiske indlæggelsesdiagnoser adiagnoser)
    df_LPR3 = df_LPR3[df_LPR3['adiagnosekode'].str.slice(1, 2).isin(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N'])]

    #så er dataframes hentet og klar til at blive justeret
    df_LPR2 = df_LPR2.rename(  # type: ignore
        columns={"datotid_indlaeggelse": "timestamp"}
    )

    df_LPR3 = df_LPR3.rename(  # type: ignore
        columns={"datotid_lpr3kontaktstart": "timestamp"}
    )
    df = pd.concat([df_LPR2, df_LPR3], ignore_index=True)

    df = df[['dw_ek_borger', 'timestamp']]

    if timestamp_as_value_col:
        df["value"] = df["timestamp"].copy()
    else:
        df["value"] = 1

    if timestamps_only:
        return df[["dw_ek_borger", "timestamp"]]

    return df

if __name__ == "__main__":
    df_pd = get_contacts_to_somatic_emergency()
    n_patients = df_pd['dw_ek_borger'].nunique()
    print(f"Antal unikke ID'er der har mindst én akut somatisk indlæggelse er: {n_patients}")
    antal_kontakter = df_pd.shape[0]
    print(f"Antal akutte somatiske indlæggelser: {antal_kontakter}")