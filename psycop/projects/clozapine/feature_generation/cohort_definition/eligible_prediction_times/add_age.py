import polars as pl

from psycop.projects.clozapine.feature_generation.cohort_definition.eligible_prediction_times.eligible_config import (
    AGE_COL_NAME,
)
from psycop.projects.clozapine.loaders.demographics import birthdays


def add_age(df: pl.DataFrame) -> pl.DataFrame:
    birthday_df = pl.from_pandas(birthdays())

    df = df.join(birthday_df, on="dw_ek_borger", how="inner")
    df = df.with_columns(
        ((pl.col("timestamp") - pl.col("date_of_birth")).dt.days()).alias(AGE_COL_NAME)
    )
    df = df.with_columns((pl.col(AGE_COL_NAME) / 365.25).alias(AGE_COL_NAME))

    return df
