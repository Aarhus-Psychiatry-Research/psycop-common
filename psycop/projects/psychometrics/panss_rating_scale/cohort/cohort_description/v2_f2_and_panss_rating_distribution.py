from datetime import datetime
from typing import Union

import pandas as pd
import polars as pl

from psycop.projects.psychometrics.loaders.diagnoses import (
    f2_disorders,
    f2_disorders_a_diagnosis,
    f2_disorders_b_diagnosis,
)
from psycop.projects.psychometrics.loaders.visits import (
    ambulatory_visits_psykometri_2025,
    physical_visits_psykometri_2025,
)
from psycop.projects.psychometrics.panss_rating_scale.cohort.outcome_specification.panss_rating_score import (
    get_panss_rating_scores,
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
    return (
        pl.from_pandas(
            ambulatory_visits_psykometri_2025(
                shak_code=6600,
                shak_sql_operator="=",
                timestamps_only=True,
                timestamp_for_output="end",
            )
        )
        .lazy()
        .rename({"timestamp": "visit_timestamp"})
    )


def load_panss_rating() -> pl.LazyFrame:
    return (
        pl.from_pandas(get_panss_rating_scores())
        .lazy()
        .select([pl.col("dw_ek_borger"), pl.col("timestamp").alias("panss_rating_timestamp")])
        .unique(subset=["dw_ek_borger", "panss_rating_timestamp"])
    )


def get_global_min_panss_timestamp() -> datetime:
    return load_panss_rating().select(pl.col("panss_rating_timestamp").min()).collect().item()


def load_f2_a_diagnoses() -> pl.LazyFrame:
    return pl.from_pandas(f2_disorders_a_diagnosis()).lazy()


def load_f2_b_diagnoses() -> pl.LazyFrame:
    return pl.from_pandas(f2_disorders_b_diagnosis()).lazy()


def load_f2_diagnoses() -> pl.LazyFrame:
    return pl.from_pandas(f2_disorders()).lazy()


# --------------------------------------------------------------
# GENERIC PIPELINES
# --------------------------------------------------------------


def inpatient_pipeline(
    adm: pl.LazyFrame, dx: pl.LazyFrame, panss: pl.LazyFrame, global_min: datetime
) -> pl.LazyFrame:
    dx_patients = dx.select("dw_ek_borger").unique()

    adm_dx = (
        adm.join(dx_patients, on="dw_ek_borger", how="inner")
        .filter(pl.col("timestamp_start") >= pl.lit(global_min))
        .with_columns(
            (
                pl.col("dw_ek_borger").cast(pl.Utf8)
                + "_"
                + pl.col("timestamp_start").dt.strftime("%Y%m%d%H%M%S")
            ).alias("admission_id")
        )
        .unique(["dw_ek_borger", "admission_id", "timestamp_start", "timestamp_end"])
    )

    joined = (
        adm_dx.join(panss, on="dw_ek_borger", how="left")
        .filter(
            (pl.col("panss_rating_timestamp") >= pl.col("timestamp_start"))
            & (pl.col("panss_rating_timestamp") <= pl.col("timestamp_end"))
        )
        .unique(["dw_ek_borger", "admission_id", "panss_rating_timestamp"])
    )

    agg = (
        joined.group_by(["dw_ek_borger", "admission_id", "timestamp_start", "timestamp_end"])
        .agg(pl.count("panss_rating_timestamp").alias("n_panss_rating"))
        .with_columns((pl.col("n_panss_rating") > 0).alias("has_panss_rating"))
    )

    base = adm_dx.with_columns(pl.lit("inpatient").alias("contact_type"))

    return base.join(
        agg, on=["dw_ek_borger", "admission_id", "timestamp_start", "timestamp_end"], how="left"
    ).with_columns(
        pl.col("n_panss_rating").fill_null(0), pl.col("has_panss_rating").fill_null(False)
    )


def outpatient_pipeline(
    op: pl.LazyFrame, dx: pl.LazyFrame, panss: pl.LazyFrame, global_min: datetime
) -> pl.LazyFrame:
    op_dx = (
        op.join(dx.select("dw_ek_borger").unique(), on="dw_ek_borger", how="inner")
        .filter(pl.col("visit_timestamp") >= pl.lit(global_min))
        .unique(["dw_ek_borger", "visit_timestamp"])
    )

    joined = (
        op_dx.join(panss, on="dw_ek_borger", how="left")
        .filter(pl.col("visit_timestamp").dt.date() == pl.col("panss_rating_timestamp").dt.date())
        .unique(["dw_ek_borger", "visit_timestamp", "panss_rating_timestamp"])
    )

    agg = (
        joined.group_by(["dw_ek_borger", "visit_timestamp"])
        .agg(pl.count("panss_rating_timestamp").alias("n_panss_rating"))
        .with_columns((pl.col("n_panss_rating") > 0).alias("has_panss_rating"))
    )

    base = op_dx.with_columns(pl.lit("outpatient").alias("contact_type"))

    return base.join(agg, on=["dw_ek_borger", "visit_timestamp"], how="left").with_columns(
        pl.col("n_panss_rating").fill_null(0), pl.col("has_panss_rating").fill_null(False)
    )


# --------------------------------------------------------------
# PIPELINES
# --------------------------------------------------------------


def inpatient_a_pipeline(
    adm: pl.LazyFrame, dx_a: pl.LazyFrame, panss: pl.LazyFrame, global_min: datetime
) -> pl.LazyFrame:
    return inpatient_pipeline(adm, dx_a, panss, global_min)


def inpatient_b_pipeline(
    adm: pl.LazyFrame, dx_b: pl.LazyFrame, panss: pl.LazyFrame, global_min: datetime
) -> pl.LazyFrame:
    return inpatient_pipeline(adm, dx_b, panss, global_min)


def outpatient_a_pipeline(
    op: pl.LazyFrame, dx_a: pl.LazyFrame, panss: pl.LazyFrame, global_min: datetime
) -> pl.LazyFrame:
    return outpatient_pipeline(op, dx_a, panss, global_min)


def outpatient_all_pipeline(
    op: pl.LazyFrame, dx_all: pl.LazyFrame, panss: pl.LazyFrame, global_min: datetime
) -> pl.LazyFrame:
    return outpatient_pipeline(op, dx_all, panss, global_min)


# --------------------------------------------------------------
# OVERVIEW
# --------------------------------------------------------------


def print_overview_tables(
    inpatient_a_df: pl.DataFrame,
    inpatient_b_df: pl.DataFrame,
    outpatient_a_df: pl.DataFrame,
    outpatient_all_df: pl.DataFrame,
) -> None:
    # ---------------- INPATIENT A ----------------
    total_a = inpatient_a_df.height
    a_with = inpatient_a_df.filter(pl.col("has_panss_rating")).height
    pct_a = (a_with / total_a * 100) if total_a else 0.0
    sum_a = inpatient_a_df["n_panss_rating"].sum()

    mean_a = (
        inpatient_a_df.filter(pl.col("n_panss_rating") > 0)["n_panss_rating"].mean()
        if inpatient_a_df.filter(pl.col("n_panss_rating") > 0).height
        else 0.0
    )

    # ---------------- INPATIENT B ----------------
    total_b = inpatient_b_df.height
    b_with = inpatient_b_df.filter(pl.col("has_panss_rating")).height
    pct_b = (b_with / total_b * 100) if total_b else 0.0
    sum_b = inpatient_b_df["n_panss_rating"].sum()

    mean_b = (
        inpatient_b_df.filter(pl.col("n_panss_rating") > 0)["n_panss_rating"].mean()
        if inpatient_b_df.filter(pl.col("n_panss_rating") > 0).height
        else 0.0
    )

    # ---------------- OUTPATIENT A ----------------
    total_out_a = outpatient_a_df.height
    out_a_with = outpatient_a_df.filter(pl.col("has_panss_rating")).height
    pct_out_a = (out_a_with / total_out_a * 100) if total_out_a else 0.0
    sum_out_a = outpatient_a_df["n_panss_rating"].sum()

    # ---------------- OUTPATIENT ALL ----------------
    total_out_all = outpatient_all_df.height
    out_all_with = outpatient_all_df.filter(pl.col("has_panss_rating")).height
    pct_out_all = (out_all_with / total_out_all * 100) if total_out_all else 0.0
    sum_out_all = outpatient_all_df["n_panss_rating"].sum()

    # ---------------- GLOBAL ----------------
    total_panss = load_panss_rating().collect().height

    # ---------------- TABLE ----------------
    overview = pl.DataFrame(
        {
            "Metric": [
                "Total inpatient admissions (F2 A)",
                "Inpatient A with ≥1 PANSS",
                "Percent inpatient A with PANSS",
                "PANSS count inpatient A",
                "Mean PANSS per inpatient A (if >0)",
                "Total inpatient admissions (F2 B)",
                "Inpatient B with ≥1 PANSS",
                "Percent inpatient B with PANSS",
                "PANSS count inpatient B",
                "Mean PANSS per inpatient B (if >0)",
                "Total outpatient visits (A)",
                "Outpatient A with PANSS",
                "Percent outpatient A with PANSS",
                "PANSS count outpatient A",
                "Total outpatient visits (All)",
                "Outpatient All with PANSS",
                "Percent outpatient All with PANSS",
                "PANSS count outpatient All",
                "Total PANSS scores (raw source)",
            ],
            "Value": [
                str(total_a),
                str(a_with),
                f"{pct_a:.1f}%",
                str(int(sum_a)),
                f"{mean_a:.2f}",
                str(total_b),
                str(b_with),
                f"{pct_b:.1f}%",
                str(int(sum_b)),
                f"{mean_b:.2f}",
                str(total_out_a),
                str(out_a_with),
                f"{pct_out_a:.1f}%",
                str(int(sum_out_a)),
                str(total_out_all),
                str(out_all_with),
                f"{pct_out_all:.1f}%",
                str(int(sum_out_all)),
                str(total_panss),
            ],
        }
    )

    print("\n=== PANSS utilisation overview ===")
    print(overview.to_pandas().to_string(index=False))


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

if __name__ == "__main__":
    adm = load_admissions()
    op = load_outpatient()
    panss = load_panss_rating()
    global_min = get_global_min_panss_timestamp()

    dx_a = load_f2_a_diagnoses()
    dx_b = load_f2_b_diagnoses()
    dx_all = load_f2_diagnoses()

    inpatient_a_df = inpatient_a_pipeline(adm, dx_a, panss, global_min).collect()
    inpatient_b_df = inpatient_b_pipeline(adm, dx_b, panss, global_min).collect()

    outpatient_a_df = outpatient_a_pipeline(op, dx_a, panss, global_min).collect()
    outpatient_all_df = outpatient_all_pipeline(op, dx_all, panss, global_min).collect()

    print_overview_tables(inpatient_a_df, inpatient_b_df, outpatient_a_df, outpatient_all_df)
