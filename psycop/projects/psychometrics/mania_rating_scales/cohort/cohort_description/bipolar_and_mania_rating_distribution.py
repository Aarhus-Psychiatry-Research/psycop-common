import polars as pl

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.projects.psychometrics.loaders.diagnoses import (
    bipolar_disorders_a_diagnosis,
    bipolar_disorders_b_diagnosis,
)
from psycop.projects.psychometrics.loaders.visits import ambulatory_visits_psykometri_2025
from psycop.projects.psychometrics.mania_rating_scales.cohort.outcome_specification.mania_rating_score import (
    get_mania_rating_scores,
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


def load_mania_rating() -> pl.LazyFrame:
    return (
        pl.from_pandas(get_mania_rating_scores())
        .lazy()
        .select([pl.col("dw_ek_borger"), pl.col("timestamp").alias("mania_rating_timestamp")])
        .unique(subset=["dw_ek_borger", "mania_rating_timestamp"])
    )


def load_bipolar_a_diagnoses() -> pl.LazyFrame:
    return pl.from_pandas(bipolar_disorders_a_diagnosis()).lazy()


def load_bipolar_b_diagnoses() -> pl.LazyFrame:
    return pl.from_pandas(bipolar_disorders_b_diagnosis()).lazy()


def load_all_bipolar_diagnoses() -> pl.LazyFrame:
    """
    Loader that returns a LazyFrame containing BOTH bipolar A
    and bipolar B diagnoses. The two source tables are loaded
    as eager DataFrames first (they are small), concatenated,
    and finally converted to a LazyFrame for downstream lazy
    processing.
    """
    df_a = pl.from_pandas(bipolar_disorders_a_diagnosis())
    df_b = pl.from_pandas(bipolar_disorders_b_diagnosis())

    return pl.concat([df_a, df_b]).lazy()


# --------------------------------------------------------------
# INPATIENT PIPELINES
# --------------------------------------------------------------
def inpatient_a_pipeline() -> pl.LazyFrame:
    adm = load_admissions()
    dx_a = load_bipolar_a_diagnoses()
    mania = load_mania_rating()

    adm_f3a = (
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
        adm_f3a.join(mania, on="dw_ek_borger", how="left")
        .filter(
            (pl.col("mania_rating_timestamp") >= pl.col("datotid_indlaeggelse"))
            & (pl.col("mania_rating_timestamp") <= pl.col("datotid_udskrivning"))
        )
        .with_columns(pl.lit("inpatient").alias("contact_type"))
    )

    agg = (
        joined.group_by(
            ["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"]
        )
        .agg(pl.col("mania_rating_timestamp").drop_nulls().count().alias("n_mania_rating"))
        .with_columns((pl.col("n_mania_rating") > 0).alias("has_mania_rating"))
    )

    base = adm_f3a.with_columns(pl.lit("inpatient").alias("contact_type"))
    return base.join(
        agg,
        on=["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"],
        how="left",
    ).with_columns(
        pl.col("n_mania_rating").fill_null(0), pl.col("has_mania_rating").fill_null(False)
    )


def inpatient_b_pipeline() -> pl.LazyFrame:
    adm = load_admissions()
    dx_b = load_bipolar_b_diagnoses()
    mania = load_mania_rating()

    adm_f3b = (
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
        adm_f3b.join(mania, on="dw_ek_borger", how="left")
        .filter(
            (pl.col("mania_rating_timestamp") >= pl.col("datotid_indlaeggelse"))
            & (pl.col("mania_rating_timestamp") <= pl.col("datotid_udskrivning"))
        )
        .with_columns(pl.lit("inpatient").alias("contact_type"))
    )

    agg = (
        joined.group_by(
            ["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"]
        )
        .agg(pl.col("mania_rating_timestamp").drop_nulls().count().alias("n_mania_rating"))
        .with_columns((pl.col("n_mania_rating") > 0).alias("has_mania_rating"))
    )

    base = adm_f3b.with_columns(pl.lit("inpatient").alias("contact_type"))
    return base.join(
        agg,
        on=["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"],
        how="left",
    ).with_columns(
        pl.col("n_mania_rating").fill_null(0), pl.col("has_mania_rating").fill_null(False)
    )


# --------------------------------------------------------------
# OUTPATIENT PIPELINE
# --------------------------------------------------------------
def outpatient_pipeline() -> pl.LazyFrame:
    """
    Outpatient pipeline that uses **only** bipolar A diagnoses.
    """
    op = load_ambulatory_visits()
    dx_a = load_bipolar_a_diagnoses()
    mania = load_mania_rating()

    op_bipolar = op.join(dx_a, on="dw_ek_borger", how="inner")

    joined = (
        op_bipolar.join(mania, on="dw_ek_borger", how="left")
        .filter(pl.col("visit_timestamp").dt.date() == pl.col("mania_rating_timestamp").dt.date())
        .with_columns(pl.lit("outpatient").alias("contact_type"))
    )

    agg = (
        joined.group_by(["dw_ek_borger", "visit_timestamp"])
        .agg(pl.count("mania_rating_timestamp").alias("n_mania_rating"))
        .with_columns((pl.col("n_mania_rating") > 0).alias("has_mania_rating"))
    )

    base = op_bipolar.with_columns(pl.lit("outpatient").alias("contact_type"))

    return base.join(agg, on=["dw_ek_borger", "visit_timestamp"], how="left").with_columns(
        pl.col("n_mania_rating").fill_null(0), pl.col("has_mania_rating").fill_null(False)
    )


# --------------------------------------------------------------
# OVERVIEW
# --------------------------------------------------------------
def print_overview_tables(
    inpatient_a_df: pl.DataFrame, inpatient_b_df: pl.DataFrame, outpatient_df: pl.DataFrame
) -> None:
    # ----- A-diagnosis -----
    total_a = inpatient_a_df.height
    a_with_mania = inpatient_a_df.filter(pl.col("has_mania_rating")).height
    pct_a_mania = a_with_mania / total_a * 100 if total_a > 0 else 0.0
    a_mania_scores = inpatient_a_df["n_mania_rating"].sum()
    a_mean_mania = (
        inpatient_a_df.filter(pl.col("n_mania_rating") > 0)["n_mania_rating"].mean()
        if inpatient_a_df.filter(pl.col("n_mania_rating") > 0).height > 0
        else 0.0
    )

    # ----- B-diagnosis -----
    total_b = inpatient_b_df.height
    b_with_mania = inpatient_b_df.filter(pl.col("has_mania_rating")).height
    pct_b_mania = b_with_mania / total_b * 100 if total_b > 0 else 0.0
    b_mania_scores = inpatient_b_df["n_mania_rating"].sum()
    b_mean_mania = (
        inpatient_b_df.filter(pl.col("n_mania_rating") > 0)["n_mania_rating"].mean()
        if inpatient_b_df.filter(pl.col("n_mania_rating") > 0).height > 0
        else 0.0
    )

    # ----- Outpatient -----
    total_out = outpatient_df.height
    out_with_mania = outpatient_df.filter(pl.col("has_mania_rating")).height
    pct_out_mania = out_with_mania / total_out * 100 if total_out > 0 else 0.0
    out_mania_scores = outpatient_df["n_mania_rating"].sum()

    # ----- Global -----
    total_ham = pl.from_pandas(get_mania_rating_scores()).height

    # ----- BUILD TABLE -----
    overview = pl.DataFrame(
        {
            "Metric": [
                "Total inpatient admissions (Bipolar A)",
                "Inpatient A admissions with ≥1 mania rating",
                "Percent A admissions with mania rating",
                "Mania rating scores in A admissions",
                "Mean mania ratings per A admission (≥1 rating)",
                "Total inpatient admissions (Bipolar B)",
                "Inpatient B admissions with ≥1 mania rating",
                "Percent B admissions with mania rating",
                "Mania rating scores in B admissions",
                "Mean mania ratings per B admission (≥1 rating)",
                "Total outpatient visits",
                "Outpatient visits with mania rating",
                "Percent outpatient visits with mania rating",
                "Mania rating scores in outpatient setting",
                # Overall
                "Total mania rating scores (all source data)",
            ],
            "Value": [
                str(total_a),
                str(a_with_mania),
                f"{pct_a_mania:.1f}%",
                str(int(a_mania_scores)),
                f"{a_mean_mania:.2f}",
                str(total_b),
                str(b_with_mania),
                f"{pct_b_mania:.1f}%",
                str(int(b_mania_scores)),
                f"{b_mean_mania:.2f}",
                str(total_out),
                str(out_with_mania),
                f"{pct_out_mania:.1f}%",
                str(int(out_mania_scores)),
                str(total_ham),
            ],
        }
    )

    print("\n=== Mania rating utilisation overview (A, B & Outpatient) ===")
    print(overview.to_pandas().to_string(index=False))


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
if __name__ == "__main__":
    inpatient_a_df = inpatient_a_pipeline().collect()
    inpatient_b_df = inpatient_b_pipeline().collect()
    outpatient_df = outpatient_pipeline().collect()
    print_overview_tables(inpatient_a_df, inpatient_b_df, outpatient_df)
