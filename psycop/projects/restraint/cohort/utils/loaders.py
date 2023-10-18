import pandas as pd

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load


def load_admissions_discharge_timestamps() -> pd.DataFrame:
    return sql_load(
        "SELECT * FROM fct.[FOR_kohorte_indhold_pt_journal_inkl_2021_feb2022]",
    )


def load_coercion_timestamps() -> pd.DataFrame:
    return sql_load(
        "SELECT * FROM fct.[FOR_tvang_alt_hele_kohorten_inkl_2021_feb2022]",
    )
