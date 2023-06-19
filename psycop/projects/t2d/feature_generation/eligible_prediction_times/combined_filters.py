import polars as pl

from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays
from psycop.projects.t2d.feature_generation.eligible_prediction_times.eligible_config import (
    AGE_COL_NAME,
)
from psycop.projects.t2d.feature_generation.eligible_prediction_times.single_filters import (
    min_age,
    min_date,
    no_incident_diabetes,
    washout_move,
    without_prevalent_diabetes,
)


def add_age(df: pl.DataFrame) -> pl.DataFrame:
    birthday_df = pl.from_pandas(birthdays())

    df = df.join(birthday_df, on="dw_ek_borger", how="inner")
    df = df.with_columns(
        ((pl.col("timestamp") - pl.col("date_of_birth")).dt.days()).alias(AGE_COL_NAME),
    )
    df = df.with_columns((pl.col(AGE_COL_NAME) / 365.25).alias(AGE_COL_NAME))

    return df


def filter_prediction_times_by_eligibility(df: pl.DataFrame) -> pl.DataFrame:
    steps = [
        min_date,
        add_age,
        min_age,
        without_prevalent_diabetes,
        no_incident_diabetes,
        washout_move,
    ]

    for step in steps:
        df = step(df)

    return df
