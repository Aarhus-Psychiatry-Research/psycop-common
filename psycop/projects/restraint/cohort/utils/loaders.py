import pandas as pd

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load


def load_admissions_discharge_timestamps() -> pd.DataFrame:
    return sql_load("SELECT * FROM fct.[FOR_kohorte_indhold_pt_journal_inkl_2021_feb2022]")


def load_coercion_timestamps() -> pd.DataFrame:
    return sql_load("SELECT * FROM fct.[FOR_tvang_alt_hele_kohorten_inkl_2021_feb2022]")


def load_prediction_timestamps_deprecated() -> pd.DataFrame:
    return sql_load(
        "SELECT dw_ek_borger, datotid_start as timestamp_admission, datotid_slut as timestamp_discharge, pred_adm_day_count, pred_time as timestamp FROM fct.psycop_coercion_outcome_timestamps"
    )
