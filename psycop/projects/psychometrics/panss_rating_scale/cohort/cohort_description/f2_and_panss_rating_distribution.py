import polars as pl

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.projects.psychometrics.loaders.diagnoses import (
    f2_disorders_a_diagnosis,
    f2_disorders_b_diagnosis,
)
from psycop.projects.psychometrics.loaders.visits import ambulatory_visits_psykometri_2025
from psycop.projects.psychometrics.panss_rating_scale.cohort.outcome_specification.panss_rating_score import (
    get_panss_rating_scores,
)


# --------------------------------------------------------------
# RAW LOADERS
# --------------------------------------------------------------
def load_admissions() -> pl.LazyFrame:
    sql = """
    SELECT
        dw_ek_borger,
        datotid_indlaeggelse,
        datotid_udskrivning
    FROM [fct].[psykometri_indlaeggelser]
    """

    df = pl.from_pandas(sql_load(sql)).lazy()

    return df


def load_ambulatory_visits() -> pl.LazyFrame:
    df_pd = ambulatory_visits_psykometri_2025(timestamps_only=True, timestamp_for_output="end")
    return (
        pl.from_pandas(df_pd)
        .select([pl.col("dw_ek_borger"), pl.col("timestamp").alias("visit_timestamp")])
        .lazy()
    )


def load_panss_rating() -> pl.LazyFrame:
    return (
        pl.from_pandas(get_panss_rating_scores())
        .lazy()
        .select([pl.col("dw_ek_borger"), pl.col("timestamp").alias("panss_rating_timestamp")])
        .unique(subset=["dw_ek_borger", "panss_rating_timestamp"])
    )


def load_f2_a_diagnoses() -> pl.LazyFrame:
    return pl.from_pandas(f2_disorders_a_diagnosis()).lazy()


def load_f2_b_diagnoses() -> pl.LazyFrame:
    return pl.from_pandas(f2_disorders_b_diagnosis()).lazy()


def load_all_f2_diagnoses() -> pl.LazyFrame:
    """
    Loader that returns a LazyFrame containing BOTH f2 A
    and f2 B diagnoses. The two source tables are loaded
    as eager DataFrames first (they are small), concatenated,
    and finally converted to a LazyFrame for downstream lazy
    processing.
    """
    df_a = pl.from_pandas(f2_disorders_a_diagnosis())
    df_b = pl.from_pandas(f2_disorders_b_diagnosis())

    return pl.concat([df_a, df_b]).lazy()


# --------------------------------------------------------------
# INPATIENT PIPELINES
# --------------------------------------------------------------
def inpatient_a_pipeline() -> pl.LazyFrame:
    adm = load_admissions()
    dx_a = load_f2_a_diagnoses()
    panss = load_panss_rating()

    adm_f2a = (
        adm.join(dx_a, on="dw_ek_borger", how="inner")
        .with_columns(
            (
                pl.col("dw_ek_borger").cast(pl.Utf8)
                + "_"
                + pl.col("datotid_indlaeggelse").dt.strftime("%Y%m%d%H%M%S")
            ).alias("admission_id")
        )
        .unique(
            subset=["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"]
        )
    )

    joined = (
        adm_f2a.join(panss, on="dw_ek_borger", how="left")
        .filter(
            (pl.col("panss_rating_timestamp") >= pl.col("datotid_indlaeggelse"))
            & (pl.col("panss_rating_timestamp") <= pl.col("datotid_udskrivning"))
        )
        .with_columns(pl.lit("inpatient").alias("contact_type"))
    )

    agg = (
        joined.group_by(
            ["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"]
        )
        .agg(pl.col("panss_rating_timestamp").drop_nulls().count().alias("n_panss_rating"))
        .with_columns((pl.col("n_panss_rating") > 0).alias("has_panss_rating"))
    )

    base = adm_f2a.with_columns(pl.lit("inpatient").alias("contact_type"))
    return base.join(
        agg,
        on=["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"],
        how="left",
    ).with_columns(
        pl.col("n_panss_rating").fill_null(0), pl.col("has_panss_rating").fill_null(False)
    )


def inpatient_b_pipeline() -> pl.LazyFrame:
    adm = load_admissions()
    dx_b = load_f2_b_diagnoses()
    panss = load_panss_rating()

    adm_f2b = (
        adm.join(dx_b, on="dw_ek_borger", how="inner")
        .with_columns(
            (
                pl.col("dw_ek_borger").cast(pl.Utf8)
                + "_"
                + pl.col("datotid_indlaeggelse").dt.strftime("%Y%m%d%H%M%S")
            ).alias("admission_id")
        )
        .unique(
            subset=["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"]
        )
    )

    joined = (
        adm_f2b.join(panss, on="dw_ek_borger", how="left")
        .filter(
            (pl.col("panss_rating_timestamp") >= pl.col("datotid_indlaeggelse"))
            & (pl.col("panss_rating_timestamp") <= pl.col("datotid_udskrivning"))
        )
        .with_columns(pl.lit("inpatient").alias("contact_type"))
    )

    agg = (
        joined.group_by(
            ["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"]
        )
        .agg(pl.col("panss_rating_timestamp").drop_nulls().count().alias("n_panss_rating"))
        .with_columns((pl.col("n_panss_rating") > 0).alias("has_panss_rating"))
    )

    base = adm_f2b.with_columns(pl.lit("inpatient").alias("contact_type"))
    return base.join(
        agg,
        on=["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"],
        how="left",
    ).with_columns(
        pl.col("n_panss_rating").fill_null(0), pl.col("has_panss_rating").fill_null(False)
    )


# --------------------------------------------------------------
# OUTPATIENT PIPELINE
# --------------------------------------------------------------
def outpatient_pipeline() -> pl.LazyFrame:
    """
    Outpatient pipeline that uses **only** F2 A diagnoses.
    """
    op = load_ambulatory_visits()
    dx_a = load_f2_a_diagnoses()
    panss = load_panss_rating()

    op_panss = op.join(dx_a, on="dw_ek_borger", how="inner")

    joined = (
        op_panss.join(panss, on="dw_ek_borger", how="left")
        .filter(pl.col("visit_timestamp").dt.date() == pl.col("panss_rating_timestamp").dt.date())
        .with_columns(pl.lit("outpatient").alias("contact_type"))
    )

    agg = (
        joined.group_by(["dw_ek_borger", "visit_timestamp"])
        .agg(pl.count("panss_rating_timestamp").alias("n_panss_rating"))
        .with_columns((pl.col("n_panss_rating") > 0).alias("has_panss_rating"))
    )

    base = op_panss.with_columns(pl.lit("outpatient").alias("contact_type"))

    return base.join(agg, on=["dw_ek_borger", "visit_timestamp"], how="left").with_columns(
        pl.col("n_panss_rating").fill_null(0), pl.col("has_panss_rating").fill_null(False)
    )


# --------------------------------------------------------------
# OVERVIEW
# --------------------------------------------------------------
def print_overview_tables(
    inpatient_a_df: pl.DataFrame, inpatient_b_df: pl.DataFrame, outpatient_df: pl.DataFrame
) -> None:
    # ----- A-diagnosis -----
    total_a = inpatient_a_df.height
    a_with_panss = inpatient_a_df.filter(pl.col("has_panss_rating")).height
    pct_a_panss = a_with_panss / total_a * 100 if total_a > 0 else 0.0
    a_panss_scores = inpatient_a_df["n_panss_rating"].sum()
    a_mean_panss = (
        inpatient_a_df.filter(pl.col("n_panss_rating") > 0)["n_panss_rating"].mean()
        if inpatient_a_df.filter(pl.col("n_panss_rating") > 0).height > 0
        else 0.0
    )

    # ----- B-diagnosis -----
    total_b = inpatient_b_df.height
    b_with_panss = inpatient_b_df.filter(pl.col("has_panss_rating")).height
    pct_b_panss = b_with_panss / total_b * 100 if total_b > 0 else 0.0
    b_panss_scores = inpatient_b_df["n_panss_rating"].sum()
    b_mean_panss = (
        inpatient_b_df.filter(pl.col("n_panss_rating") > 0)["n_panss_rating"].mean()
        if inpatient_b_df.filter(pl.col("n_panss_rating") > 0).height > 0
        else 0.0
    )

    # ----- Outpatient -----
    total_out = outpatient_df.height
    out_with_panss = outpatient_df.filter(pl.col("has_panss_rating")).height
    pct_out_panss = out_with_panss / total_out * 100 if total_out > 0 else 0.0
    out_panss_scores = outpatient_df["n_panss_rating"].sum()

    # ----- Global -----
    total_ham = pl.from_pandas(get_panss_rating_scores()).height

    # ----- BUILD TABLE -----
    overview = pl.DataFrame(
        {
            "Metric": [
                "Total inpatient admissions (F2-diagnosis A)",
                "Inpatient A admissions with ≥1 panss rating",
                "Percent A admissions with panss rating",
                "Panss rating scores in A admissions",
                "Mean panss ratings per A admission (≥1 rating)",
                "Total inpatient admissions (F2-diagnosis B)",
                "Inpatient B admissions with ≥1 panss rating",
                "Percent B admissions with panss rating",
                "Panss rating scores in B admissions",
                "Mean panss ratings per B admission (≥1 rating)",
                "Total outpatient visits",
                "Outpatient visits with panss rating",
                "Percent outpatient visits with panss rating",
                "Panss rating scores in outpatient setting",
                # Overall
                "Total panss rating scores (all source data)",
            ],
            "Value": [
                str(total_a),
                str(a_with_panss),
                f"{pct_a_panss:.1f}%",
                str(int(a_panss_scores)),
                f"{a_mean_panss:.2f}",
                str(total_b),
                str(b_with_panss),
                f"{pct_b_panss:.1f}%",
                str(int(b_panss_scores)),
                f"{b_mean_panss:.2f}",
                str(total_out),
                str(out_with_panss),
                f"{pct_out_panss:.1f}%",
                str(int(out_panss_scores)),
                str(total_out),
            ],
        }
    )

    print("\n=== PANSS rating utilisation overview (A, B & Outpatient) ===")
    print(overview.to_pandas().to_string(index=False))


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
if __name__ == "__main__":
    inpatient_a_df = inpatient_a_pipeline().collect()
    inpatient_b_df = inpatient_b_pipeline().collect()
    outpatient_df = outpatient_pipeline().collect()
    print_overview_tables(inpatient_a_df, inpatient_b_df, outpatient_df)
