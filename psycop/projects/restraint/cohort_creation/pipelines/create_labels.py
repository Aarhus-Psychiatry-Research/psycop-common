"""
This script creates the labels for the psycop coercion project.

Labels: Kig to dage frem
- Hierarchy of coercion instances
"""

from datetime import date

import pandas as pd

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.common.global_utils.sql.writer import write_df_to_sql
from psycop.projects.restraint.cohort_creation.utils.cohort_hyperparameters import (
    create_binary_and_categorical_labels_with_lookahead,
)

df_cohort = sql_load(
    "SELECT * FROM fct.[psycop_coercion_cohort_with_all_days_without_labels_feb2022]",
)

lookahead_days = 2


# First coercion to pred_time
df_cohort["diff_first_coercion"] = pd.to_datetime(
    df_cohort["datotid_start_sei"],
) - pd.to_datetime(df_cohort["pred_time"])

# First mechanical restraint to pred_time
df_cohort["diff_first_mechanical_restraint"] = pd.to_datetime(
    df_cohort["first_mechanical_restraint"],
) - pd.to_datetime(df_cohort["pred_time"])

# First forced medication to pred_time
df_cohort["diff_first_forced_medication"] = pd.to_datetime(
    df_cohort["first_forced_medication"],
) - pd.to_datetime(df_cohort["pred_time"])

# First manual restraint to pred_time
df_cohort["diff_first_manual_restraint"] = pd.to_datetime(
    df_cohort["first_manual_restraint"],
) - pd.to_datetime(df_cohort["pred_time"])


# apply create_labels function to data
df_cohort = create_binary_and_categorical_labels_with_lookahead(
    df_cohort,
    lookahead_days=lookahead_days,
)


# only include admission days at which coercion has not happened yet
df_cohort = df_cohort[df_cohort["include_pred_time"] == 1]


# Rename columns, drop irrelevant columns, and reset index
df_cohort = (
    df_cohort.rename(
        columns={
            "datotid_start": "timestamp_admission",
            "datotid_slut": "timestamp_discharge",
            "datotid_start_sei": "timestamp_outcome",
            "pred_time": "timestamp",
        },
    )
    .drop(
        columns=(
            [
                "behandlingsomraade",
                "first_mechanical_restraint",
                "first_forced_medication",
                "first_manual_restraint",
                "include_pred_time",
                "diff_first_coercion",
                "diff_first_mechanical_restraint",
                "diff_first_forced_medication",
                "diff_first_manual_restraint",
            ]
        ),
    )
    .reset_index(drop=True)
)

# add admission id
df_cohort.insert(
    loc=0,
    column="adm_id",
    value=df_cohort.dw_ek_borger.astype(str)
    + "-"
    + df_cohort.timestamp_admission.astype(str),
)


# write csv named with today's date
today = date.today().strftime("%d%m%y")
df_cohort.to_csv(
    f"psycop_coercion_within_{lookahead_days}_days_feb2022_run_{today}.csv",
)

# Write to sql database
write_df_to_sql(
    df=df_cohort,
    table_name=f"psycop_coercion_within_{lookahead_days}_days_feb2022",
    if_exists="replace",
    rows_per_chunk=5000,
)
