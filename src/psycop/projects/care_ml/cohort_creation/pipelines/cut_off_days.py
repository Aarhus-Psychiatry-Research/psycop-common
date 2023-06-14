"""In this script, we cut off days in the cohort that are after the mean+std admission duration"""

from datetime import date

import pandas as pd
from care_ml.cohort_creation.utils.cohort_hyperparameters import cut_off_prediction_days
from care_ml.cohort_creation.utils.utils import cut_off_check
from psycop.common.feature_generation.loaders.raw.sql_load import sql_load

# ---------------------------------
# LOAD DATA
# ---------------------------------

# Load cohort
df_cohort = sql_load(query="SELECT * FROM fct.[psycop_coercion_within_2_days_feb2022]")

# Load train (to find cut-off)
df_train = pd.read_parquet(
    path="E:/shared_resources/coercion/feature_sets/psycop_coercion_adminsignbe_features_2023_03_31_13_39/psycop_coercion_adminsignbe_features_2023_03_31_13_39_train.parquet",
)

df_val = pd.read_parquet(
    path="E:/shared_resources/coercion/feature_sets/psycop_coercion_adminsignbe_features_2023_03_31_13_39/psycop_coercion_adminsignbe_features_2023_03_31_13_39_val.parquet",
)

df_train = pd.concat([df_train, df_val])

# ---------------------------------
# Train: Admission durations
# ---------------------------------

df_adm_grain = df_train[
    [
        "adm_id",
        "dw_ek_borger",
        "timestamp_admission",
        "timestamp_discharge",
        "outcome_timestamp",
    ]
].drop_duplicates(keep="first")

# calculate adm duration
df_adm_grain["adm_duration"] = (
    df_adm_grain["timestamp_discharge"] - df_adm_grain["timestamp_admission"]
)
df_train["adm_duration"] = (
    df_train["timestamp_discharge"] - df_train["timestamp_admission"]
)

# ---------------------------------
# Train: Cut-off definition
# ---------------------------------

# How many days and coericon instances will we lose?

cut_off_check(df_cohort, df_adm_grain, 1)
cut_off_check(df_cohort, df_adm_grain, 2)
cut_off_check(df_cohort, df_adm_grain, 3)

# ---------------------------------
# Cohort: Cut off days after cut-off
# ---------------------------------

cut_off = df_adm_grain["adm_duration"].mean() + df_adm_grain["adm_duration"].std()

df_cohort_exclude_days_after_cut_off = cut_off_prediction_days(df_cohort, cut_off)  # type: ignore

# ---------------------------------
# WRITE CSV
# ---------------------------------

# write csv named with today's date
today = date.today().strftime("%d%m%y")
lookahead_days = 2
df_cohort_exclude_days_after_cut_off.to_csv(
    f"psycop_coercion_within_{lookahead_days}_days_feb2022_exclude_days_after_cut_off_run_{today}.csv",
)
