"""
Hamilton distribution-pipeline with both F3 A- og F3 B-diagnoses.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Union

import pandas as pd
import polars as pl

from psycop.projects.psychometrics.hamilton.cohort.outcome_specification.hamilton_score import (
    get_hamilton_scores,
)
from psycop.projects.psychometrics.loaders.utils import parse_diagnosegruppestreng_to_diagnoses
from psycop.projects.psychometrics.loaders.visits import (
    ambulatory_visits_psykometri_2025,
    physical_visits_psykometri_2025,
)


# ----------------------------------------------------------------------
# RAW LOADERS
# ----------------------------------------------------------------------
def admissions_start(
    n_rows: Union[int, None] = None, return_value_as_visit_length_days: Union[bool, None] = False
) -> pd.DataFrame:
    """Load admissions."""
    return physical_visits_psykometri_2025(
        timestamp_for_output="start",
        visit_types=["admissions"],
        return_value_as_visit_length_days=return_value_as_visit_length_days,
        n_rows=n_rows,
        shak_code=6600,
        shak_sql_operator="=",
    )


def admissions_end(
    n_rows: Union[int, None] = None, return_value_as_visit_length_days: Union[bool, None] = False
) -> pd.DataFrame:
    """Load admissions."""
    return physical_visits_psykometri_2025(
        timestamp_for_output="end",
        visit_types=["admissions"],
        return_value_as_visit_length_days=return_value_as_visit_length_days,
        n_rows=n_rows,
        shak_code=6600,
        shak_sql_operator="=",
    )


def load_admissions() -> pl.LazyFrame:
    df_start = pl.from_pandas(admissions_start()).lazy()
    df_end = pl.from_pandas(admissions_end()).lazy()

    # rename timestamps
    df_start = df_start.rename({"timestamp": "timestamp_start"})
    df_end = df_end.rename({"timestamp": "timestamp_end"})

    # 🔥 DROP value early to avoid duplication chaos
    df_start = df_start.drop(["value"])
    df_end = df_end.drop(["value"])

    # sort (required for asof join)
    df_start = df_start.sort(["dw_ek_borger", "timestamp_start"])
    df_end = df_end.sort(["dw_ek_borger", "timestamp_end"])

    # asof join: match closest end AFTER start
    df = df_start.join_asof(
        df_end,
        left_on="timestamp_start",
        right_on="timestamp_end",
        by="dw_ek_borger",
        strategy="forward",
    )

    # final cleanup: ensure valid intervals
    df = df.filter(
        pl.col("timestamp_end").is_not_null()
        & (pl.col("timestamp_end") >= pl.col("timestamp_start"))
    )

    return df


def load_outpatient() -> pl.LazyFrame:
    df_pd = ambulatory_visits_psykometri_2025(
        shak_code=6600, shak_sql_operator="=", timestamps_only=True, timestamp_for_output="end"
    )

    return pl.from_pandas(df_pd).lazy().rename({"timestamp": "visit_timestamp"})


def load_hamilton() -> pl.LazyFrame:
    return (
        pl.from_pandas(get_hamilton_scores())
        .lazy()
        .select([pl.col("dw_ek_borger"), pl.col("timestamp").alias("hamilton_rating_timestamp")])
        .unique(subset=["dw_ek_borger", "hamilton_rating_timestamp"])
    )


# ----------------------------------------------------------------------
# F3 DIAGNOSIS LOADERS — now resolved PER VISIT ROW, not per patient.
#
# Each visit row already carries its own `diagnosegruppestreng`. We parse
# that string with `parse_diagnosegruppestreng_to_diagnoses` and keep only
# the rows that actually contain an F3 code, instead of flagging every
# admission/visit of a patient who has an F3 diagnosis *somewhere* in
# their history.
# ----------------------------------------------------------------------
def _row_has_f3(
    diagnosegruppestreng: Union[str, None], diagnosis_type: Union[Literal["A", "B"], None]
) -> bool:
    if diagnosegruppestreng is None:
        return False
    codes = parse_diagnosegruppestreng_to_diagnoses(
        diagnosegruppestreng, diagnosis_type=diagnosis_type
    )
    if not codes:
        return False
    return any(c.lower().startswith("f3") for c in codes)


def _f3_diagnosis_visits(
    timestamp_for_output: Literal["start", "end"],
    visit_types: Sequence[Literal["admissions", "ambulatory_visits", "emergency_visits"]],
    diagnosis_type: Union[Literal["A", "B"], None],
) -> pd.DataFrame:
    """Load visits (with diagnosegruppestreng kept) and filter to rows that
    themselves carry an F3 diagnosis of the requested type (A/B/either)."""
    df = physical_visits_psykometri_2025(
        timestamp_for_output=timestamp_for_output,
        visit_types=visit_types,
        shak_code=6600,
        shak_sql_operator="=",
        keep_diagnosegruppestreng=True,
    )

    mask = df["diagnosegruppestreng"].apply(lambda x: _row_has_f3(x, diagnosis_type=diagnosis_type))

    return df[mask].reset_index(drop=True)


def load_f3_a_admissions_diagnoses() -> pl.LazyFrame:
    """Admissions whose own contact carries an F3 A-diagnosis."""
    df = _f3_diagnosis_visits(
        timestamp_for_output="start", visit_types=["admissions"], diagnosis_type="A"
    )
    return (
        pl.from_pandas(df)
        .lazy()
        .select(["dw_ek_borger", "timestamp"])
        .rename({"timestamp": "timestamp_start"})
        .unique()
    )


def load_f3_b_admissions_diagnoses() -> pl.LazyFrame:
    """Admissions whose own contact carries an F3 B-diagnosis."""
    df = _f3_diagnosis_visits(
        timestamp_for_output="start", visit_types=["admissions"], diagnosis_type="B"
    )
    return (
        pl.from_pandas(df)
        .lazy()
        .select(["dw_ek_borger", "timestamp"])
        .rename({"timestamp": "timestamp_start"})
        .unique()
    )


def load_f3_a_outpatient_diagnoses() -> pl.LazyFrame:
    """Outpatient visits whose own contact carries an F3 A-diagnosis."""
    df = _f3_diagnosis_visits(
        timestamp_for_output="end", visit_types=["ambulatory_visits"], diagnosis_type="A"
    )
    return (
        pl.from_pandas(df)
        .lazy()
        .select(["dw_ek_borger", "timestamp"])
        .rename({"timestamp": "visit_timestamp"})
        .unique()
    )


def load_f3_any_outpatient_diagnoses() -> pl.LazyFrame:
    """Outpatient visits whose own contact carries any F3 diagnosis (A or B)."""
    df = _f3_diagnosis_visits(
        timestamp_for_output="end", visit_types=["ambulatory_visits"], diagnosis_type=None
    )
    return (
        pl.from_pandas(df)
        .lazy()
        .select(["dw_ek_borger", "timestamp"])
        .rename({"timestamp": "visit_timestamp"})
        .unique()
    )


# ----------------------------------------------------------------------
# GENERIC PIPELINES
# ----------------------------------------------------------------------
def inpatient_pipeline(
    admissions: pl.LazyFrame, diagnoses: pl.LazyFrame, hamilton: pl.LazyFrame, global_min: datetime
) -> pl.LazyFrame:
    """`diagnoses` must be a LazyFrame of (dw_ek_borger, timestamp_start)
    identifying the *specific* admissions that carry the F3 diagnosis —
    not just patients who have it anywhere in their history."""

    adm_dx = (
        admissions.join(diagnoses, on=["dw_ek_borger", "timestamp_start"], how="inner")
        .filter(pl.col("timestamp_start") >= pl.lit(global_min))
        .with_columns(
            (
                pl.col("dw_ek_borger").cast(pl.Utf8)
                + "_"
                + pl.col("timestamp_start").dt.strftime("%Y%m%d%H%M%S")
            ).alias("admission_id")
        )
        .unique(subset=["dw_ek_borger", "admission_id", "timestamp_start", "timestamp_end"])
    )

    joined = (
        adm_dx.join(hamilton, on="dw_ek_borger", how="left")
        .filter(
            (pl.col("hamilton_rating_timestamp") >= pl.col("timestamp_start"))
            & (pl.col("hamilton_rating_timestamp") <= pl.col("timestamp_end"))
        )
        .with_columns(pl.lit("inpatient").alias("contact_type"))
        .unique(subset=["dw_ek_borger", "admission_id", "hamilton_rating_timestamp"])
    )

    agg = (
        joined.group_by(["dw_ek_borger", "admission_id", "timestamp_start", "timestamp_end"])
        .agg(pl.n_unique("hamilton_rating_timestamp").alias("n_hamilton"))
        .with_columns((pl.col("n_hamilton") > 0).alias("has_hamilton"))
    )

    base = adm_dx.with_columns(pl.lit("inpatient").alias("contact_type"))

    return base.join(
        agg, on=["dw_ek_borger", "admission_id", "timestamp_start", "timestamp_end"], how="left"
    ).with_columns(pl.col("n_hamilton").fill_null(0), pl.col("has_hamilton").fill_null(False))


def outpatient_pipeline(
    outpatient: pl.LazyFrame, diagnoses: pl.LazyFrame, hamilton: pl.LazyFrame, global_min: datetime
) -> pl.LazyFrame:
    """`diagnoses` must be a LazyFrame of (dw_ek_borger, visit_timestamp)
    identifying the *specific* outpatient visits that carry the F3
    diagnosis — not just patients who have it anywhere in their history."""

    op_dx = (
        outpatient.join(diagnoses, on=["dw_ek_borger", "visit_timestamp"], how="inner")
        .filter(pl.col("visit_timestamp") >= pl.lit(global_min))
        .unique(subset=["dw_ek_borger", "visit_timestamp"])
    )

    joined = (
        op_dx.join(hamilton, on="dw_ek_borger", how="left")
        .filter(
            pl.col("visit_timestamp").dt.date() == pl.col("hamilton_rating_timestamp").dt.date()
        )
        .with_columns(pl.lit("outpatient").alias("contact_type"))
        .unique(subset=["dw_ek_borger", "visit_timestamp", "hamilton_rating_timestamp"])
    )

    agg = (
        joined.group_by(["dw_ek_borger", "visit_timestamp"])
        .agg(pl.n_unique("hamilton_rating_timestamp").alias("n_hamilton"))
        .with_columns((pl.col("n_hamilton") > 0).alias("has_hamilton"))
    )

    base = op_dx.with_columns(pl.lit("outpatient").alias("contact_type"))

    return base.join(agg, on=["dw_ek_borger", "visit_timestamp"], how="left").with_columns(
        pl.col("n_hamilton").fill_null(0), pl.col("has_hamilton").fill_null(False)
    )


def print_overview_tables(
    inpatient_a_df: pl.DataFrame,
    inpatient_b_df: pl.DataFrame,
    outpatient_a_df: pl.DataFrame,
    outpatient_all_df: pl.DataFrame,
) -> None:
    tot_a = inpatient_a_df.height
    a_with = inpatient_a_df.filter(pl.col("has_hamilton")).height
    pct_a = (a_with / tot_a * 100) if tot_a else 0.0
    ham_a = inpatient_a_df["n_hamilton"].sum()
    mean_a = (
        inpatient_a_df.filter(pl.col("n_hamilton") > 0)["n_hamilton"].mean()
        if inpatient_a_df.filter(pl.col("n_hamilton") > 0).height
        else 0.0
    )

    tot_b = inpatient_b_df.height
    b_with = inpatient_b_df.filter(pl.col("has_hamilton")).height
    pct_b = (b_with / tot_b * 100) if tot_b else 0.0
    ham_b = inpatient_b_df["n_hamilton"].sum()
    mean_b = (
        inpatient_b_df.filter(pl.col("n_hamilton") > 0)["n_hamilton"].mean()
        if inpatient_b_df.filter(pl.col("n_hamilton") > 0).height
        else 0.0
    )

    tot_out_a = outpatient_a_df.height
    out_a_with = outpatient_a_df.filter(pl.col("has_hamilton")).height
    pct_out_a = (out_a_with / tot_out_a * 100) if tot_out_a else 0.0
    ham_out_a = outpatient_a_df["n_hamilton"].sum()

    tot_out_b = outpatient_all_df.height
    out_b_with = outpatient_all_df.filter(pl.col("has_hamilton")).height
    pct_out_b = (out_b_with / tot_out_b * 100) if tot_out_b else 0.0
    ham_out_b = outpatient_all_df["n_hamilton"].sum()

    total_ham_all = pl.from_pandas(get_hamilton_scores()).height

    overview = pl.DataFrame(
        {
            "Metric": [
                "Total admissions (F3 A)",
                "Admissions (F3 A) with ≥1 Hamilton score",
                "Percent admissions (F3 A) with Hamilton",
                "Hamilton scores in F3 A admissions",
                "Mean Hamilton scores per admission (F3 A)",
                "Total admissions (F3 B)",
                "Admissions (F3 B) with ≥1 Hamilton score",
                "Percent admissions (F3 B) with Hamilton",
                "Hamilton scores in F3 B admissions",
                "Mean Hamilton scores per admission (F3 B)",
                "Total outpatient visits (F3 A)",
                "Outpatient visits (F3 A) with ≥1 Hamilton score",
                "Percent outpatient visits (F3 A) with Hamilton",
                "Hamilton scores in outpatient F3 A",
                "Total outpatient visits (F3 any)",
                "Outpatient visits (F3 any) with ≥1 Hamilton score",
                "Percent outpatient visits (F3 any) with Hamilton",
                "Hamilton scores in outpatient F3 any",
                "Total Hamilton scores (all source data)",
            ],
            "Value": [
                str(tot_a),
                str(a_with),
                f"{pct_a:.1f}%",
                str(int(ham_a)),
                f"{mean_a:.2f}",
                str(tot_b),
                str(b_with),
                f"{pct_b:.1f}%",
                str(int(ham_b)),
                f"{mean_b:.2f}",
                str(tot_out_a),
                str(out_a_with),
                f"{pct_out_a:.1f}%",
                str(int(ham_out_a)),
                str(tot_out_b),
                str(out_b_with),
                f"{pct_out_b:.1f}%",
                str(int(ham_out_b)),
                str(total_ham_all),
            ],
        }
    )
    print("\n=== Hamilton Overview (F3 A & F3 B) ===")
    print(overview.to_pandas().to_string(index=False))


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # expensive loads only once
    admissions = load_admissions()
    outpatient = load_outpatient()
    hamilton = load_hamilton()

    global_min = hamilton.select(pl.col("hamilton_rating_timestamp").min()).collect().item()

    dx_a_admissions = load_f3_a_admissions_diagnoses()
    dx_b_admissions = load_f3_b_admissions_diagnoses()
    dx_a_outpatient = load_f3_a_outpatient_diagnoses()
    dx_any_outpatient = load_f3_any_outpatient_diagnoses()

    inpatient_a_df = inpatient_pipeline(
        admissions=admissions, diagnoses=dx_a_admissions, hamilton=hamilton, global_min=global_min
    ).collect()

    inpatient_b_df = inpatient_pipeline(
        admissions=admissions, diagnoses=dx_b_admissions, hamilton=hamilton, global_min=global_min
    ).collect()

    outpatient_a_df = outpatient_pipeline(
        outpatient=outpatient, diagnoses=dx_a_outpatient, hamilton=hamilton, global_min=global_min
    ).collect()

    outpatient_all_df = outpatient_pipeline(
        outpatient=outpatient, diagnoses=dx_any_outpatient, hamilton=hamilton, global_min=global_min
    ).collect()

    print_overview_tables(inpatient_a_df, inpatient_b_df, outpatient_a_df, outpatient_all_df)
