import polars as pl
from psycop.feature_generation.application_modules.filter_prediction_times import (
    PredictionTimeFilterer,
)
from psycop.feature_generation.loaders.raw.load_moves import (
    load_move_into_rm_for_exclusion,
)
from psycop.projects.t2d.feature_generation.eligible_prediction_times.eligible_config import (
    AGE_COL_NAME,
    MIN_AGE,
    MIN_DATE,
)
from psycop.projects.t2d.feature_generation.eligible_prediction_times.tooling import (
    add_stepdelta_from_df,
    add_stepdelta_manual,
)
from psycop.projects.t2d.feature_generation.outcome_specification.combined import (
    get_first_diabetes_indicator,
)
from psycop.projects.t2d.feature_generation.outcome_specification.lab_results import (
    get_first_diabetes_lab_result_above_threshold,
)


def min_date(df: pl.DataFrame) -> pl.DataFrame:
    after_df = df.filter(pl.col("timestamp") > MIN_DATE)
    add_stepdelta_from_df(step_name="min_date", before_df=df, after_df=after_df)
    return after_df


def min_age(df: pl.DataFrame) -> pl.DataFrame:
    after_df = df.filter(pl.col(AGE_COL_NAME) >= MIN_AGE)
    add_stepdelta_from_df(step_name="min_age", before_df=df, after_df=after_df)
    return after_df


def without_prevalent_diabetes(df: pl.DataFrame) -> pl.DataFrame:
    first_diabetes_indicator = pl.from_pandas(get_first_diabetes_indicator())

    indicator_before_min_date = first_diabetes_indicator.filter(
        pl.col("timestamp") < MIN_DATE,
    )

    prediction_times_from_patients_with_diabetes = df.join(
        indicator_before_min_date,
        on="dw_ek_borger",
        how="inner",
    )

    hit_indicator = prediction_times_from_patients_with_diabetes.groupby(
        "source",
    ).count()

    for indicator in hit_indicator.rows(named=True):
        add_stepdelta_manual(
            step_name=indicator["source"],
            n_before=df.shape[0],
            n_after=df.shape[0] - indicator["count"],
        )

    no_prevalent_diabetes = df.join(
        prediction_times_from_patients_with_diabetes,
        on="dw_ek_borger",
        how="anti",
    )

    add_stepdelta_from_df(
        step_name="No prevalent diabetes",
        before_df=df,
        after_df=no_prevalent_diabetes,
    )

    return no_prevalent_diabetes


def no_incident_diabetes(df: pl.DataFrame) -> pl.DataFrame:
    results_above_threshold = pl.from_pandas(
        get_first_diabetes_lab_result_above_threshold(),
    )

    contacts_with_hba1c = df.join(
        results_above_threshold,
        on="dw_ek_borger",
        how="left",
        suffix="_result",
    )

    after_incident_diabetes = contacts_with_hba1c.filter(
        pl.col("timestamp") > pl.col("timestamp_result"),
    )

    not_after_incident_diabetes = contacts_with_hba1c.join(
        after_incident_diabetes,
        on="dw_ek_borger",
        how="anti",
    )

    add_stepdelta_from_df(
        step_name="no_incident_diabetes",
        before_df=df,
        after_df=not_after_incident_diabetes,
    )

    return not_after_incident_diabetes


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

    add_stepdelta_from_df(
        step_name="washout_move",
        before_df=df,
        after_df=not_within_two_years_from_move,
    )

    return not_within_two_years_from_move
