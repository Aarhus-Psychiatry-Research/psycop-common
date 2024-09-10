import polars as pl

from psycop.common.feature_generation.loaders_2024.demographics import birthdays
from psycop.projects.t2d_extended.cohort_definition.eligible_prediction_times.eligible_config import (
    AGE_COL_NAME,
)


def add_age(df: pl.DataFrame) -> pl.DataFrame:
    birthday_df = pl.from_pandas(
        birthdays(sql_cmd_postfix="AND left()")
    )  # TD: How can we filter on SHAK? Is that even present?

    df = df.join(birthday_df, on="dw_ek_borger", how="inner")
    df = df.with_columns(
        ((pl.col("timestamp") - pl.col("date_of_birth")).dt.days()).alias(AGE_COL_NAME)
    )
    df = df.with_columns((pl.col(AGE_COL_NAME) / 365.25).alias(AGE_COL_NAME))

    return df
