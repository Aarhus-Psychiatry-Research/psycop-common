import pandas as pd

from psycop.projects.clozapine.loaders.lab_results import p_clozapine


def find_plasma_clozapine_between_2013_2014() -> pd.DataFrame:
    p_clozapin_df = p_clozapine()

    filtered_p_clozapine_df = p_clozapin_df[
        (p_clozapin_df["timestamp"] >= "2013-01-01") & (p_clozapin_df["timestamp"] <= "2013-12-31")
    ]

    p_clozapin_df_only_timestamp = filtered_p_clozapine_df[["dw_ek_borger", "timestamp"]]

    return p_clozapin_df_only_timestamp
