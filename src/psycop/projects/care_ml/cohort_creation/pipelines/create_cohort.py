"""
This script creates the cohort for the psycop coercion project.
"""

from datetime import date

import numpy as np
import pandas as pd
from care_ml.cohort_creation.utils.cohort_hyperparameters import (
    exclude_prior_outcome_with_lookbehind,
)
from care_ml.cohort_creation.utils.utils import (
    concat_readmissions,
    first_coercion_within_admission,
    unpack_adm_days,
)
from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop_ml_utils.sql.writer import write_df_to_sql

# load data
df_adm = sql_load(
    "SELECT * FROM fct.[FOR_kohorte_indhold_pt_journal_inkl_2021_feb2022]",
)  # only includes admissions in psychiatry (shak code starts with 6600)
df_coercion = sql_load(
    "SELECT * FROM fct.[FOR_tvang_alt_hele_kohorten_inkl_2021_feb2022]",
)  # includes coercion in both psychiatry and somatic


# ADMISSIONS DATA
# only keep admissions (not ambulatory visits)
df_adm = df_adm[df_adm["pt_type"] == "Indlagt"]

# only keep age >= 18 at the start of contact
df_adm = df_adm[df_adm["alder_start"] >= 18]

# only keep admissions after January 1st 2015 (so we can use use lookbehind windows of two years)
df_adm = df_adm[df_adm["datotid_start"] >= "2015-01-01"]

# only keep relevant columns
df_adm = df_adm[["dw_ek_borger", "datotid_start", "datotid_slut"]]

# COERCION DATA
# only include target coercion types: manual restraint, forced medication, and mechanical restraint, excluding voluntary mechanical restraint (i.e., "fastholdelse", "beroligende medicin", and "bæltefiksering", excluding "frivillig bæltefiksering")
df_coercion = df_coercion[
    (
        (df_coercion.typetekst_sei == "Bælte")
        & (df_coercion.begrundtekst_sei != "Frivillig bæltefiksering")
    )
    | (df_coercion.typetekst_sei == "Fastholden")
    | (df_coercion.typetekst_sei == "Beroligende medicin")
]

# only keep relevant columns
df_coercion = df_coercion[
    ["dw_ek_borger", "datotid_start_sei", "typetekst_sei", "behandlingsomraade"]
]

# sort based on patient and start of admission
df_adm = df_adm.sort_values(["dw_ek_borger", "datotid_start"])

# group by patient
df_patients = df_adm.groupby("dw_ek_borger")

# list of dfs; one for each patient
df_patients_list = [df_patients.get_group(key) for key in df_patients.groups]

# concatenate dataframes for individual patients
df_adm = pd.concat([concat_readmissions(patient) for patient in df_patients_list])

# for all patients, join all instances of coercion onto all admissions
df_cohort = df_adm.merge(df_coercion, how="left", on="dw_ek_borger")


# exclude admission if there has been an instance of coercion between 0 and 365 days before admission start (including 0 and 365)
df_excluded_admissions = exclude_prior_outcome_with_lookbehind(
    df_cohort,
    lookbehind=365,
    col_admission_start="datotid_start",
    col_outcome_start="datotid_start_sei",
)[["dw_ek_borger", "datotid_start"]]

# remove duplicate rows, so we have one row per admission (instead of multiple rows for admissions with multiple coercion instances)
df_excluded_admissions = df_excluded_admissions.drop_duplicates(keep="first")

# outer join of admissions and excluded admissions with and indicator column ("_merge") denoting whether and observation occurs in both datasets
df_cohort = df_cohort.merge(
    df_excluded_admissions,
    how="outer",
    on=["dw_ek_borger", "datotid_start"],
    indicator=True,
)

# exclude rows that are in both datasets (i.e., exclude admissions in "df_excluded_admissions")
df_cohort = df_cohort.loc[df_cohort["_merge"] != "both"]


# only keep instances of coercion that occured during the particular admission
df_cohort_with_coercion = df_cohort[
    (df_cohort["datotid_start_sei"] > df_cohort["datotid_start"])
    & (df_cohort["datotid_start_sei"] < df_cohort["datotid_slut"])
]

# keep first time of coercion for each admission
# group by admission
df_admissions = df_cohort_with_coercion.groupby(["dw_ek_borger", "datotid_start"])
df_admissions_list = [df_admissions.get_group(key) for key in df_admissions.groups]


df_cohort_with_coercion = pd.concat(
    [first_coercion_within_admission(admission) for admission in df_admissions_list],
)

# remove irrelevant columns from df_cohort, drop duplicates
df_cohort = df_cohort[
    ["dw_ek_borger", "datotid_start", "datotid_slut"]
].drop_duplicates()


# merge with df_cohort_coercion
df_cohort = df_cohort.merge(
    df_cohort_with_coercion,
    how="left",
    on=["dw_ek_borger", "datotid_start", "datotid_slut"],
)


# we exclude admissions with na discharge day and discharge day > 2021-11-22 due to legal restrictions
df_cohort = df_cohort[
    (df_cohort.datotid_slut.notna()) & (df_cohort.datotid_slut <= "2021-11-22")
]


# for each admission, we want to make a prediction every day


# Apply the function unpack_adm_days to all patients
df_cohort = pd.concat([unpack_adm_days(idx, row) for idx, row in df_cohort.iterrows()])  # type: ignore


# Create include_pred_time_column (pred times were coercion hasn't happened yet or no coercion in the admission)
df_cohort["include_pred_time"] = np.where(
    (df_cohort.pred_time < df_cohort.datotid_start_sei)
    | (df_cohort.datotid_start_sei.isna()),
    1,
    0,
)


# load admission data again
df_adm = sql_load(
    "SELECT * FROM fct.[FOR_kohorte_indhold_pt_journal_inkl_2021_feb2022]",
)  # only includes admissions in psychiatry (shak code starts with 6600)

# only keep admission contacts
df_adm = df_adm[df_adm["pt_type"] == "Indlagt"]

# only keep age >= 18 at the start of contact
df_adm = df_adm[df_adm["alder_start"] >= 18]

# only keep admissions after January 1st 2015 (so we can use use lookbehind windows of two years)
df_adm = df_adm[df_adm["datotid_start"] >= "2015-01-01"]

# only keep relevant columns
df_adm = df_adm[["dw_ek_borger", "datotid_start", "shakkode_ansvarlig"]]

# left join df_adm on df_cohort
df_cohort = df_cohort.merge(
    df_adm,
    how="left",
    on=["dw_ek_borger", "datotid_start"],
)

# remove admissions in the department of forensic psychiatry (shak code 6600021 and 6600310)
df_cohort = df_cohort[
    (df_cohort["shakkode_ansvarlig"] != "6600310")
    & (df_cohort["shakkode_ansvarlig"] != "6600021")
]


# remove coercion in somatics
df_cohort = df_cohort[df_cohort["behandlingsomraade"] != "Somatikken"]


# write csv with today's date
today = date.today().strftime("%d%m%y")
df_cohort.to_csv(f"cohort_{today}.csv")

# Write to sql database
write_df_to_sql(
    df=df_cohort,
    table_name="psycop_coercion_cohort_with_all_days_without_labels_feb2022",
    if_exists="replace",
    rows_per_chunk=5000,
)
