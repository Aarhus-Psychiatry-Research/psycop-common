import polars as pl

from psycop.projects.psychometrics.loaders.diagnoses import f2_disorders


def add_only_patients_with_f2_diagnosis() -> pl.DataFrame:
    df_f2_disorders = pl.from_pandas(f2_disorders())

    f2_disorders_df_only_time_and_borger = df_f2_disorders.select(["dw_ek_borger", "timestamp"])

    return f2_disorders_df_only_time_and_borger
