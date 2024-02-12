import polars as pl

from psycop.common.feature_generation.loaders.raw.load_diagnoses import schizophrenia


def add_only_patients_with_schizophrenia() -> pl.DataFrame:
    schizophrenia_df = pl.from_pandas(schizophrenia())

    schizophrenia_df_only_time_and_borger = schizophrenia_df.select(["dw_ek_borger","timestamp"])

    return schizophrenia_df_only_time_and_borger

