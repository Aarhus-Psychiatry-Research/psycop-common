"""
Hamilton distribution-pipeline with both F3 A- og F3 B-diagnoses.
"""

import polars as pl

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.projects.psychometrics.hamilton.cohort.outcome_specification.hamilton_score import (
    get_hamilton_scores,
)
from psycop.projects.psychometrics.loaders.diagnoses import (
    f3_disorders_a_diagnosis,
    f3_disorders_b_diagnosis,
)


# ----------------------------------------------------------------------
# RAW LOADERS
# ----------------------------------------------------------------------
def load_admissions() -> pl.LazyFrame:
    sql = """
    SELECT
        dw_ek_borger,
        datotid_indlaeggelse,
        datotid_udskrivning
    FROM [fct].[psykometri_indlaeggelser]
    """
    return pl.from_pandas(sql_load(sql)).lazy()


def load_outpatient_visits() -> pl.LazyFrame:
    sql = """
    SELECT
        dw_ek_borger,
        datotid_start
    FROM [fct].[psykometri_besoeg]
    """
    return pl.from_pandas(sql_load(sql)).rename({"datotid_start": "visit_timestamp"}).lazy()


def load_hamilton() -> pl.LazyFrame:
    return (
        pl.from_pandas(get_hamilton_scores())
        .lazy()
        .select([pl.col("dw_ek_borger"), pl.col("timestamp").alias("hamilton_timestamp")])
    )


def load_f3_a_diagnoses() -> pl.LazyFrame:
    return pl.from_pandas(f3_disorders_a_diagnosis()).lazy()


def load_f3_b_diagnoses() -> pl.LazyFrame:
    return pl.from_pandas(f3_disorders_b_diagnosis()).lazy()


# ----------------------------------------------------------------------
# INPATIENT PIPELINES (A & B)
# ----------------------------------------------------------------------
def inpatient_a_pipeline() -> pl.LazyFrame:
    adm = load_admissions()
    dx_a = load_f3_a_diagnoses()
    ham = load_hamilton()

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
        adm_f3a.join(ham, on="dw_ek_borger", how="left")
        .filter(
            (pl.col("hamilton_timestamp") >= pl.col("datotid_indlaeggelse"))
            & (pl.col("hamilton_timestamp") <= pl.col("datotid_udskrivning"))
        )
        .with_columns(pl.lit("inpatient").alias("contact_type"))
        .unique(subset=["dw_ek_borger", "admission_id", "hamilton_timestamp"])
    )

    agg = (
        joined.group_by(
            ["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"]
        )
        .agg(pl.n_unique("hamilton_timestamp").alias("n_hamilton"))
        .with_columns((pl.col("n_hamilton") > 0).alias("has_hamilton"))
    )

    base = adm_f3a.with_columns(pl.lit("inpatient").alias("contact_type"))

    return base.join(
        agg,
        on=["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"],
        how="left",
    ).with_columns(pl.col("n_hamilton").fill_null(0), pl.col("has_hamilton").fill_null(False))


def inpatient_b_pipeline() -> pl.LazyFrame:
    adm = load_admissions()
    dx_b = load_f3_b_diagnoses()
    ham = load_hamilton()

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
        adm_f3b.join(ham, on="dw_ek_borger", how="left")
        .filter(
            (pl.col("hamilton_timestamp") >= pl.col("datotid_indlaeggelse"))
            & (pl.col("hamilton_timestamp") <= pl.col("datotid_udskrivning"))
        )
        .with_columns(pl.lit("inpatient").alias("contact_type"))
        .unique(subset=["dw_ek_borger", "admission_id", "hamilton_timestamp"])
    )

    agg = (
        joined.group_by(
            ["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"]
        )
        .agg(pl.n_unique("hamilton_timestamp").alias("n_hamilton"))
        .with_columns((pl.col("n_hamilton") > 0).alias("has_hamilton"))
    )

    base = adm_f3b.with_columns(pl.lit("inpatient").alias("contact_type"))

    return base.join(
        agg,
        on=["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"],
        how="left",
    ).with_columns(pl.col("n_hamilton").fill_null(0), pl.col("has_hamilton").fill_null(False))


# ----------------------------------------------------------------------
# OUTPATIENT PIPELINES (A & B)
# ----------------------------------------------------------------------
def outpatient_a_pipeline() -> pl.LazyFrame:
    op = load_outpatient_visits()
    dx_a = load_f3_a_diagnoses()
    ham = load_hamilton()

    op_f3a = op.join(dx_a, on="dw_ek_borger", how="inner")
    joined = (
        op_f3a.join(ham, on="dw_ek_borger", how="left")
        .filter(pl.col("visit_timestamp").dt.date() == pl.col("hamilton_timestamp").dt.date())
        .with_columns(pl.lit("outpatient").alias("contact_type"))
        .unique(subset=["dw_ek_borger", "visit_timestamp", "hamilton_timestamp"])
    )
    agg = (
        joined.group_by(["dw_ek_borger", "visit_timestamp"])
        .agg(pl.n_unique("hamilton_timestamp").alias("n_hamilton"))
        .with_columns((pl.col("n_hamilton") > 0).alias("has_hamilton"))
    )
    base = op_f3a.with_columns(pl.lit("outpatient").alias("contact_type"))
    return base.join(agg, on=["dw_ek_borger", "visit_timestamp"], how="left").with_columns(
        pl.col("n_hamilton").fill_null(0), pl.col("has_hamilton").fill_null(False)
    )


def outpatient_b_pipeline() -> pl.LazyFrame:
    op = load_outpatient_visits()
    dx_b = load_f3_b_diagnoses()
    ham = load_hamilton()

    op_f3b = op.join(dx_b, on="dw_ek_borger", how="inner")
    joined = (
        op_f3b.join(ham, on="dw_ek_borger", how="left")
        .filter(pl.col("visit_timestamp").dt.date() == pl.col("hamilton_timestamp").dt.date())
        .with_columns(pl.lit("outpatient").alias("contact_type"))
        .unique(subset=["dw_ek_borger", "visit_timestamp", "hamilton_timestamp"])
    )
    agg = (
        joined.group_by(["dw_ek_borger", "visit_timestamp"])
        .agg(pl.n_unique("hamilton_timestamp").alias("n_hamilton"))
        .with_columns((pl.col("n_hamilton") > 0).alias("has_hamilton"))
    )
    base = op_f3b.with_columns(pl.lit("outpatient").alias("contact_type"))
    return base.join(agg, on=["dw_ek_borger", "visit_timestamp"], how="left").with_columns(
        pl.col("n_hamilton").fill_null(0), pl.col("has_hamilton").fill_null(False)
    )


# ----------------------------------------------------------------------
# OVERVIEW
# ----------------------------------------------------------------------
def print_overview_tables(
    inpatient_a_df: pl.DataFrame,
    inpatient_b_df: pl.DataFrame,
    outpatient_a_df: pl.DataFrame,
    outpatient_b_df: pl.DataFrame,
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

    tot_out_b = outpatient_b_df.height
    out_b_with = outpatient_b_df.filter(pl.col("has_hamilton")).height
    pct_out_b = (out_b_with / tot_out_b * 100) if tot_out_b else 0.0
    ham_out_b = outpatient_b_df["n_hamilton"].sum()

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
                "Total outpatient visits (F3 B)",
                "Outpatient visits (F3 B) with ≥1 Hamilton score",
                "Percent outpatient visits (F3 B) with Hamilton",
                "Hamilton scores in outpatient F3B",
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
    print("\n=== Hamilton Utilisation Overview (F3 A & F3 B) ===")
    print(overview.to_pandas().to_string(index=False))


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    inpatient_a_df = inpatient_a_pipeline().collect()
    inpatient_b_df = inpatient_b_pipeline().collect()
    outpatient_a_df = outpatient_a_pipeline().collect()
    outpatient_b_df = outpatient_b_pipeline().collect()

    print_overview_tables(inpatient_a_df, inpatient_b_df, outpatient_a_df, outpatient_b_df)
