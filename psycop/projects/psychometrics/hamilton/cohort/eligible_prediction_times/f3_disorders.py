import polars as pl

from psycop.projects.psychometrics.loaders.diagnoses import f3_disorders


def add_only_patients_with_f3_diagnosis() -> pl.DataFrame:
    df_f3_disorders = pl.from_pandas(f3_disorders())

    f3_disorders_df_only_time_and_borger = df_f3_disorders.select(["dw_ek_borger", "timestamp"])

    return f3_disorders_df_only_time_and_borger
