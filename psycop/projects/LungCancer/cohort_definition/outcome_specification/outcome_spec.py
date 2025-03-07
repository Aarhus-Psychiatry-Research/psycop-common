"""Script for creating cancer cohort.
Currently, the output of this file is not used as the cohort, though it might be necessary to use this approach for later work (e.g. when distinguishing between different types of cancer)
"""

from pathlib import Path

import pandas as pd

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load

CANCER_DATA_DIR = Path(r"E:\shared_resources") / "cancer"

# LPR3, both in and outpatient
df_lpr3_preproc = sql_load("SELECT * FROM [fct].FOR_LPR3kontakter_psyk_somatik_inkl_2021")[
    ["dw_ek_borger", "datotid_lpr3kontaktstart", "adiagnosekode", "shakkode_lpr3kontaktansvarlig"]
]

df_lpr3_preproc = df_lpr3_preproc.rename(
    columns={
        "datotid_lpr3kontaktstart": "datotid_start",
        "shakkode_lpr3kontaktansvarlig": "shakafskode",
    }
)

# LPR2
# inpatient
df_lpr2_inp_preproc = sql_load("SELECT * FROM [fct].FOR_indlaeggelser_psyk_somatik_LPR2_inkl_2021")[
    ["dw_ek_borger", "adiagnosekode", "datotid_indlaeggelse", "shakKode_kontaktansvarlig"]
]

df_lpr2_inp_preproc = df_lpr2_inp_preproc.rename(
    columns={"datotid_indlaeggelse": "datotid_start", "shakKode_kontaktansvarlig": "shakafskode"}
)

# outpatient
df_lpr2_outp_preproc = sql_load("SELECT * FROM [fct].FOR_besoeg_psyk_somatik_LPR2_inkl_2021")[
    ["dw_ek_borger", "diagnoseKode", "datotid_start", "shakafskode"]
]

df_lpr2_outp_preproc = df_lpr2_outp_preproc.rename(columns={"diagnoseKode": "adiagnosekode"})

# Combine all
all_visits_combined = pd.concat(
    [df_lpr3_preproc, df_lpr2_inp_preproc, df_lpr2_outp_preproc], ignore_index=True
)

# extract first visit to psych in RM
df_first_psych_visit = (
    all_visits_combined[all_visits_combined["shakafskode"].str.startswith("6600")]
    .groupby(["dw_ek_borger"])["datotid_start"]
    .min()
    .to_frame()
    .reset_index()
)
df_first_psych_visit = df_first_psych_visit.rename(
    columns={"datotid_start": "datotid_first_psych_visit"}
)


# Extract cancer diagnosis
DIAGNOSIS_CODE = "DC"

df_cancer_visits = all_visits_combined[
    all_visits_combined["adiagnosekode"].str.startswith(DIAGNOSIS_CODE, na=False)
]

# only include patients that have been diagnosed with cancer after their first visit to psychiatry and after 2013
df_cancer_visits_ = df_cancer_visits.merge(df_first_psych_visit, on="dw_ek_borger")

df_cancer_visits_after_first_psych_visit = df_cancer_visits_[
    (df_cancer_visits_["datotid_start"] > df_cancer_visits_["datotid_first_psych_visit"])
    & (df_cancer_visits_["datotid_start"] > "2013-01-01")
]


# Saving cohort (in E:/shared_resources/cancer)
df_cancer_visits_after_first_psych_visit.to_csv(f"{CANCER_DATA_DIR}/cancer_cohort.csv")
