"""Loaders for cancer outcomes."""
#from __future__ import annotations

from typing import Literal

import pandas as pd

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.common.feature_generation.loaders.raw.load_diagnoses import from_contacts
from psycop.common.feature_generation.utils import data_loaders


# LPR3, both in and outpatient
df_lpr3_preproc = sql_load(
    "SELECT * FROM [fct].FOR_LPR3kontakter_psyk_somatik_inkl_2021",
)[["dw_ek_borger", "datotid_lpr3kontaktstart", "adiagnosekode", "shakkode_lpr3kontaktansvarlig"]]

df_lpr3_preproc.rename(columns={
            "datotid_lpr3kontaktstart": "datotid_start",
            "shakkode_lpr3kontaktansvarlig": "shakafskode",
            }, inplace = True
        )


# LPR2
#inpatient
df_lpr2_inp_preproc = sql_load(
    "SELECT * FROM [fct].FOR_indlaeggelser_psyk_somatik_LPR2_inkl_2021",
)[["dw_ek_borger", "adiagnosekode", "datotid_indlaeggelse", "shakKode_kontaktansvarlig"]]

df_lpr2_inp_preproc.rename(columns={
            "datotid_indlaeggelse": "datotid_start",
            "shakKode_kontaktansvarlig": "shakafskode",
            }, inplace = True
        )

#outpatient
df_lpr2_outp_preproc = sql_load(
    "SELECT * FROM [fct].FOR_besoeg_psyk_somatik_LPR2_inkl_2021",
)[["dw_ek_borger", "diagnoseKode", "datotid_start", "shakafskode"]]

df_lpr2_outp_preproc.rename(columns={
            "diagnoseKode": "adiagnosekode",
            }, inplace = True
        )


# Combine all
all_visits_combined = pd.concat([df_lpr3_preproc, df_lpr2_inp_preproc, df_lpr2_outp_preproc])

df_first_psych_visit = 

