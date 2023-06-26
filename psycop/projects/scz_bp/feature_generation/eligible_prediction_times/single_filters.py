import polars as pl

from psycop.common.feature_generation.application_modules.filter_prediction_times import (
    PredictionTimeFilterer,
)
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays
from psycop.common.feature_generation.loaders.raw.load_moves import (
    load_move_into_rm_for_exclusion,
)
from psycop.common.feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_eligible_config import (
    AGE_COL_NAME,
    MAX_AGE,
    MIN_AGE,
    MIN_DATE,
    N_DAYS_WASHIN,
)


def min_date(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("timestamp") > MIN_DATE)


def min_age(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col(AGE_COL_NAME) >= MIN_AGE)


def max_age(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col(AGE_COL_NAME) <= MAX_AGE)


def time_from_first_visit(df: pl.DataFrame) -> pl.DataFrame:
    # get first visit
    first_visit = pl.from_pandas(
        physical_visits_to_psychiatry(
            n_rows=None,
            timestamps_only=False,
            return_value_as_visit_length_days=False,
            timestamp_for_output="start",
        )
    ).groupby("dw_ek_borger").agg(pl.col("timestamp").min().alias("first_visit"))

    # left join first visit to df
    df = df.join(first_visit, on="dw_ek_borger", how="left")
    # exclude those with less than N_DAYS_WASHIN days from first visit

    ## check what format timestamp is in
    df = df.filter(
        (pl.col("timestamp") - pl.col("first_visit")) >= N_DAYS_WASHIN,
    )
    return df


def washout_move(df: pl.DataFrame) -> pl.DataFrame:
    not_within_two_years_from_move = pl.from_pandas(
        PredictionTimeFilterer(
            prediction_times_df=df.to_pandas(),
            entity_id_col_name="dw_ek_borger",
            quarantine_timestamps_df=load_move_into_rm_for_exclusion(),
            quarantine_interval_days=730,
            timestamp_col_name="timestamp",
        ).run_filter(),
    )
    return not_within_two_years_from_move 


def add_age(df: pl.DataFrame) -> pl.DataFrame:
    birthday_df = pl.from_pandas(birthdays())

    df = df.join(birthday_df, on="dw_ek_borger", how="inner")
    df = df.with_columns(
        ((pl.col("timestamp") - pl.col("date_of_birth")).dt.days()).alias(AGE_COL_NAME),
    )
    df = df.with_columns((pl.col(AGE_COL_NAME) / 365.25).alias(AGE_COL_NAME))

    return df

