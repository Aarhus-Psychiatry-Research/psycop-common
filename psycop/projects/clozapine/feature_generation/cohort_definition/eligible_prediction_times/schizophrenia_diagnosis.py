import polars as pl

from psycop.projects.clozapine.loaders.diagnoses import schizoaffective, schizophrenia


def add_only_patients_with_schizo() -> pl.DataFrame:
    schizophrenia_df = pl.from_pandas(schizophrenia())

    schizoaffective_df = pl.from_pandas(schizoaffective())

    combined_df = pl.concat([schizophrenia_df, schizoaffective_df])

    schizo_df_only_time_and_borger = combined_df.select(["dw_ek_borger", "timestamp"])

    return schizo_df_only_time_and_borger
