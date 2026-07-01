from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Union

import pandas as pd
import polars as pl

from psycop.projects.psychometrics.loaders.utils import parse_diagnosegruppestreng_to_diagnoses
from psycop.projects.psychometrics.loaders.visits import (
    ambulatory_visits_psykometri_2025,
    physical_visits_psykometri_2025,
)
from psycop.projects.psychometrics.mania_rating_scales.cohort.outcome_specification.mania_rating_score import (
    get_mania_rating_scores,
)

# --------------------------------------------------------------
# RAW LOADERS
# --------------------------------------------------------------


def admissions_start(
    n_rows: Union[int, None] = None, return_value_as_visit_length_days: Union[bool, None] = False
) -> pd.DataFrame:
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

    df_start = df_start.rename({"timestamp": "timestamp_start"}).drop(["value"])
    df_end = df_end.rename({"timestamp": "timestamp_end"}).drop(["value"])

    df_start = df_start.sort(["dw_ek_borger", "timestamp_start"])
    df_end = df_end.sort(["dw_ek_borger", "timestamp_end"])

    df = df_start.join_asof(
        df_end,
        left_on="timestamp_start",
        right_on="timestamp_end",
        by="dw_ek_borger",
        strategy="forward",
    )

    return df.filter(
        pl.col("timestamp_end").is_not_null()
        & (pl.col("timestamp_end") >= pl.col("timestamp_start"))
    )


def load_outpatient() -> pl.LazyFrame:
    df_pd = ambulatory_visits_psykometri_2025(
        shak_code=6600, shak_sql_operator="=", timestamps_only=True, timestamp_for_output="end"
    )
    return pl.from_pandas(df_pd).lazy().rename({"timestamp": "visit_timestamp"})


def load_mas() -> pl.LazyFrame:
    return (
        pl.from_pandas(get_mania_rating_scores())
        .lazy()
        .select([pl.col("dw_ek_borger"), pl.col("timestamp").alias("mania_rating_timestamp")])
        .unique(subset=["dw_ek_borger", "mania_rating_timestamp"])
    )


def get_global_min_mania_timestamp() -> datetime:
    return load_mas().select(pl.col("mania_rating_timestamp").min()).collect().item()


# --------------------------------------------------------------
# BIPOLAR DIAGNOSIS LOADERS — visit-level, not patient-level.
#
# Each row from physical_visits already carries its own
# `diagnosegruppestreng`. We parse that string with
# `parse_diagnosegruppestreng_to_diagnoses` and keep only the
# rows that actually contain an F30/F31 code of the requested
# type (A/B/either), so no admission/outpatient visit is
# included just because the patient has bipolar *somewhere else*
# in their history.
# --------------------------------------------------------------

_BIPOLAR_PREFIXES = ("f30", "f31")


def _row_has_bipolar(
    diagnosegruppestreng: Union[str, None], diagnosis_type: Union[Literal["A", "B"], None]
) -> bool:
    if diagnosegruppestreng is None:
        return False
    codes = parse_diagnosegruppestreng_to_diagnoses(
        diagnosegruppestreng, diagnosis_type=diagnosis_type
    )
    if not codes:
        return False
    return any(c.lower().startswith(prefix) for c in codes for prefix in _BIPOLAR_PREFIXES)


def _bipolar_diagnosis_visits(
    timestamp_for_output: Literal["start", "end"],
    visit_types: Sequence[Literal["admissions", "ambulatory_visits", "emergency_visits"]],
    diagnosis_type: Union[Literal["A", "B"], None],
) -> pd.DataFrame:
    """Load visits with diagnosegruppestreng and keep only rows that
    carry an F30/F31 (bipolar/manic) diagnosis of the requested type."""
    df = physical_visits_psykometri_2025(
        timestamp_for_output=timestamp_for_output,
        visit_types=visit_types,
        shak_code=6600,
        shak_sql_operator="=",
        keep_diagnosegruppestreng=True,
    )
    mask = df["diagnosegruppestreng"].apply(
        lambda x: _row_has_bipolar(x, diagnosis_type=diagnosis_type)
    )
    return df[mask].reset_index(drop=True)


def load_bipolar_a_admissions_diagnoses() -> pl.LazyFrame:
    """Admissions whose own contact carries a bipolar A-diagnosis (F30/F31 A)."""
    df = _bipolar_diagnosis_visits(
        timestamp_for_output="start", visit_types=["admissions"], diagnosis_type="A"
    )
    return (
        pl.from_pandas(df)
        .lazy()
        .select(["dw_ek_borger", "timestamp"])
        .rename({"timestamp": "timestamp_start"})
        .unique()
    )


def load_bipolar_b_admissions_diagnoses() -> pl.LazyFrame:
    """Admissions whose own contact carries a bipolar B-diagnosis (F30/F31 B)."""
    df = _bipolar_diagnosis_visits(
        timestamp_for_output="start", visit_types=["admissions"], diagnosis_type="B"
    )
    return (
        pl.from_pandas(df)
        .lazy()
        .select(["dw_ek_borger", "timestamp"])
        .rename({"timestamp": "timestamp_start"})
        .unique()
    )


def load_bipolar_a_outpatient_diagnoses() -> pl.LazyFrame:
    """Outpatient visits whose own contact carries a bipolar A-diagnosis."""
    df = _bipolar_diagnosis_visits(
        timestamp_for_output="end", visit_types=["ambulatory_visits"], diagnosis_type="A"
    )
    return (
        pl.from_pandas(df)
        .lazy()
        .select(["dw_ek_borger", "timestamp"])
        .rename({"timestamp": "visit_timestamp"})
        .unique()
    )


def load_bipolar_any_outpatient_diagnoses() -> pl.LazyFrame:
    """Outpatient visits whose own contact carries any bipolar diagnosis (A or B)."""
    df = _bipolar_diagnosis_visits(
        timestamp_for_output="end", visit_types=["ambulatory_visits"], diagnosis_type=None
    )
    return (
        pl.from_pandas(df)
        .lazy()
        .select(["dw_ek_borger", "timestamp"])
        .rename({"timestamp": "visit_timestamp"})
        .unique()
    )


# --------------------------------------------------------------
# GENERIC PIPELINES
# --------------------------------------------------------------


def inpatient_pipeline(
    admissions: pl.LazyFrame, diagnoses: pl.LazyFrame, mas: pl.LazyFrame, global_min: datetime
) -> pl.LazyFrame:
    """`diagnoses` must be (dw_ek_borger, timestamp_start) for the specific
    admissions that carry the bipolar diagnosis on that contact."""

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
        adm_dx.join(mas, on="dw_ek_borger", how="left")
        .filter(
            (pl.col("mania_rating_timestamp") >= pl.col("timestamp_start"))
            & (pl.col("mania_rating_timestamp") <= pl.col("timestamp_end"))
        )
        .with_columns(pl.lit("inpatient").alias("contact_type"))
    )

    agg = (
        joined.group_by(["dw_ek_borger", "admission_id", "timestamp_start", "timestamp_end"])
        .agg(pl.col("mania_rating_timestamp").count().alias("n_mania_rating"))
        .with_columns((pl.col("n_mania_rating") > 0).alias("has_mania_rating"))
    )

    base = adm_dx.with_columns(pl.lit("inpatient").alias("contact_type"))

    return base.join(
        agg, on=["dw_ek_borger", "admission_id", "timestamp_start", "timestamp_end"], how="left"
    ).with_columns(
        pl.col("n_mania_rating").fill_null(0), pl.col("has_mania_rating").fill_null(False)
    )


def outpatient_pipeline(
    outpatient: pl.LazyFrame, diagnoses: pl.LazyFrame, mas: pl.LazyFrame, global_min: datetime
) -> pl.LazyFrame:
    """`diagnoses` must be (dw_ek_borger, visit_timestamp) for the specific
    outpatient visits that carry the bipolar diagnosis on that contact."""

    op_dx = (
        outpatient.join(diagnoses, on=["dw_ek_borger", "visit_timestamp"], how="inner")
        .filter(pl.col("visit_timestamp") >= pl.lit(global_min))
        .unique(subset=["dw_ek_borger", "visit_timestamp"])
    )

    joined = (
        op_dx.join(mas, on="dw_ek_borger", how="left")
        .filter(pl.col("visit_timestamp").dt.date() == pl.col("mania_rating_timestamp").dt.date())
        .with_columns(pl.lit("outpatient").alias("contact_type"))
    )

    agg = (
        joined.group_by(["dw_ek_borger", "visit_timestamp"])
        .agg(pl.count("mania_rating_timestamp").alias("n_mania_rating"))
        .with_columns((pl.col("n_mania_rating") > 0).alias("has_mania_rating"))
    )

    base = op_dx.with_columns(pl.lit("outpatient").alias("contact_type"))

    return base.join(agg, on=["dw_ek_borger", "visit_timestamp"], how="left").with_columns(
        pl.col("n_mania_rating").fill_null(0), pl.col("has_mania_rating").fill_null(False)
    )


# --------------------------------------------------------------
# OVERVIEW
# --------------------------------------------------------------


def print_overview_tables(
    inpatient_a_df: pl.DataFrame,
    inpatient_b_df: pl.DataFrame,
    outpatient_df: pl.DataFrame,
    outpatient_all_df: pl.DataFrame,
) -> None:
    total_a = inpatient_a_df.height
    a_with = inpatient_a_df.filter(pl.col("has_mania_rating")).height
    pct_a = a_with / total_a * 100 if total_a > 0 else 0.0
    a_sum = inpatient_a_df["n_mania_rating"].sum()
    a_mean = (
        inpatient_a_df.filter(pl.col("n_mania_rating") > 0)["n_mania_rating"].mean()
        if inpatient_a_df.filter(pl.col("n_mania_rating") > 0).height > 0
        else 0.0
    )

    total_b = inpatient_b_df.height
    b_with = inpatient_b_df.filter(pl.col("has_mania_rating")).height
    pct_b = b_with / total_b * 100 if total_b > 0 else 0.0
    b_sum = inpatient_b_df["n_mania_rating"].sum()
    b_mean = (
        inpatient_b_df.filter(pl.col("n_mania_rating") > 0)["n_mania_rating"].mean()
        if inpatient_b_df.filter(pl.col("n_mania_rating") > 0).height > 0
        else 0.0
    )

    total_out = outpatient_df.height
    out_with = outpatient_df.filter(pl.col("has_mania_rating")).height
    pct_out = out_with / total_out * 100 if total_out > 0 else 0.0
    out_sum = outpatient_df["n_mania_rating"].sum()

    total_all = outpatient_all_df.height
    all_with = outpatient_all_df.filter(pl.col("has_mania_rating")).height
    pct_all = all_with / total_all * 100 if total_all > 0 else 0.0
    all_sum = outpatient_all_df["n_mania_rating"].sum()

    total_mania = pl.from_pandas(get_mania_rating_scores()).height

    overview = pl.DataFrame(
        {
            "Metric": [
                "Total inpatient A",
                "Inpatient A with mania",
                "Percent A with mania",
                "Mania A total",
                "Mean mania A",
                "Total inpatient B",
                "Inpatient B with mania",
                "Percent B with mania",
                "Mania B total",
                "Mean mania B",
                "Total outpatient (A)",
                "Outpatient A with mania",
                "Percent outpatient A",
                "Mania outpatient A",
                "Total outpatient (all)",
                "Outpatient all with mania",
                "Percent outpatient all",
                "Mania outpatient all",
                "Total mania scores",
            ],
            "Value": [
                str(total_a),
                str(a_with),
                f"{pct_a:.1f}%",
                str(int(a_sum)),
                f"{a_mean:.2f}",
                str(total_b),
                str(b_with),
                f"{pct_b:.1f}%",
                str(int(b_sum)),
                f"{b_mean:.2f}",
                str(total_out),
                str(out_with),
                f"{pct_out:.1f}%",
                str(int(out_sum)),
                str(total_all),
                str(all_with),
                f"{pct_all:.1f}%",
                str(int(all_sum)),
                str(total_mania),
            ],
        }
    )

    print("\n=== Mania rating utilisation overview ===")
    print(overview.to_pandas().to_string(index=False))


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

if __name__ == "__main__":
    # Expensive loads only once
    admissions = load_admissions()
    outpatient = load_outpatient()
    mas = load_mas()
    global_min = get_global_min_mania_timestamp()

    dx_a_admissions = load_bipolar_a_admissions_diagnoses()
    dx_b_admissions = load_bipolar_b_admissions_diagnoses()
    dx_a_outpatient = load_bipolar_a_outpatient_diagnoses()
    dx_any_outpatient = load_bipolar_any_outpatient_diagnoses()

    inpatient_a_df = inpatient_pipeline(
        admissions=admissions, diagnoses=dx_a_admissions, mas=mas, global_min=global_min
    ).collect()

    inpatient_b_df = inpatient_pipeline(
        admissions=admissions, diagnoses=dx_b_admissions, mas=mas, global_min=global_min
    ).collect()

    outpatient_df = outpatient_pipeline(
        outpatient=outpatient, diagnoses=dx_a_outpatient, mas=mas, global_min=global_min
    ).collect()

    outpatient_all_df = outpatient_pipeline(
        outpatient=outpatient, diagnoses=dx_any_outpatient, mas=mas, global_min=global_min
    ).collect()

    print_overview_tables(inpatient_a_df, inpatient_b_df, outpatient_df, outpatient_all_df)
