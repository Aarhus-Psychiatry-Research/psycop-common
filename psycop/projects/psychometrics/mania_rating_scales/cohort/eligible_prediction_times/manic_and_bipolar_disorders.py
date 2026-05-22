import polars as pl

from psycop.projects.psychometrics.loaders.diagnoses import manic_and_bipolar


def add_only_patients_with_manic_and_bipolar_diagnosis() -> pl.DataFrame:
    df_manic_and_bipolar_disorders = pl.from_pandas(manic_and_bipolar())

    manic_and_bipolar_disorders_df_only_time_and_borger = df_manic_and_bipolar_disorders.select(
        ["dw_ek_borger", "timestamp"]
    )

    return manic_and_bipolar_disorders_df_only_time_and_borger
