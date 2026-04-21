import polars as pl

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.projects.psychometrics.hamilton.cohort.outcome_specification.hamilton_score import (
    get_hamilton_scores,
)
from psycop.projects.psychometrics.loaders.diagnoses import f3_disorders_a_diagnosis

# ----------------------------
# RAW LOADERS
# ----------------------------


def load_admissions() -> pl.LazyFrame:
    sql = """
    SELECT
        dw_ek_borger,
        datotid_indlaeggelse,
        datotid_udskrivning
    FROM [fct].[psykometri_indlaeggelser]
    """
    return pl.from_pandas(sql_load(sql)).lazy()


def load_outpatient() -> pl.LazyFrame:
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


# ----------------------------
# INPATIENT PIPELINE
# ----------------------------


def inpatient_pipeline() -> pl.LazyFrame:
    adm = load_admissions()
    dx = load_f3_a_diagnoses()
    ham = load_hamilton()

    adm_f3a = adm.join(dx, on="dw_ek_borger", how="inner")

    joined = (
        adm_f3a.join(ham, on="dw_ek_borger", how="left")
        .filter(
            (pl.col("hamilton_timestamp") >= pl.col("datotid_indlaeggelse"))
            & (pl.col("hamilton_timestamp") <= pl.col("datotid_udskrivning"))
        )
        .with_columns(
            (
                pl.col("dw_ek_borger").cast(pl.Utf8)
                + "_"
                + pl.col("datotid_indlaeggelse").dt.strftime("%Y%m%d%H%M%S")
            ).alias("admission_id"),
            pl.lit("inpatient").alias("contact_type"),
        )
    )

    agg = (
        joined.group_by(
            ["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"]
        )
        .agg(pl.count("hamilton_timestamp").alias("n_hamilton"))
        .with_columns((pl.col("n_hamilton") > 0).alias("has_hamilton"))
    )

    base = adm_f3a.with_columns(
        (
            pl.col("dw_ek_borger").cast(pl.Utf8)
            + "_"
            + pl.col("datotid_indlaeggelse").dt.strftime("%Y%m%d%H%M%S")
        ).alias("admission_id"),
        pl.lit("inpatient").alias("contact_type"),
    )

    return base.join(
        agg,
        on=["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"],
        how="left",
    ).with_columns(pl.col("n_hamilton").fill_null(0), pl.col("has_hamilton").fill_null(False))


# ----------------------------
# OUTPATIENT PIPELINE
# ----------------------------


def outpatient_pipeline() -> pl.LazyFrame:
    op = load_outpatient()
    dx = load_f3_a_diagnoses()
    ham = load_hamilton()

    op_f3a = op.join(dx, on="dw_ek_borger", how="inner")

    joined = (
        op_f3a.join(ham, on="dw_ek_borger", how="left")
        .filter(pl.col("visit_timestamp").dt.date() == pl.col("hamilton_timestamp").dt.date())
        .with_columns(pl.lit("outpatient").alias("contact_type"))
    )

    agg = (
        joined.group_by(["dw_ek_borger", "visit_timestamp"])
        .agg(pl.count("hamilton_timestamp").alias("n_hamilton"))
        .with_columns((pl.col("n_hamilton") > 0).alias("has_hamilton"))
    )

    base = op_f3a.with_columns(pl.lit("outpatient").alias("contact_type"))

    return base.join(agg, on=["dw_ek_borger", "visit_timestamp"], how="left").with_columns(
        pl.col("n_hamilton").fill_null(0), pl.col("has_hamilton").fill_null(False)
    )


# ----------------------------
# FINAL TABLE
# ----------------------------


def print_overview_tables(inpatient_df: pl.DataFrame, outpatient_df: pl.DataFrame) -> None:
    """
    Prints overview statistics for F3(A) cohort:

    - Total number of contacts
    - Contacts with ≥1 Hamilton score
    - Percentage with Hamilton
    - Total Hamilton scores (inpatient vs outpatient)
    - Mean number of Hamilton scores per inpatient admission
    """

    # ----------------------------
    # INPATIENT METRICS
    # ----------------------------
    total_admissions = inpatient_df.height

    admissions_with_ham = inpatient_df.filter(pl.col("has_hamilton")).height

    pct_admissions_with_ham = (
        admissions_with_ham / total_admissions * 100 if total_admissions > 0 else 0.0
    )

    ham_in_admissions = inpatient_df["n_hamilton"].sum()

    mean_ham_per_admission = inpatient_df["n_hamilton"].mean() if total_admissions > 0 else 0.0

    # ----------------------------
    # OUTPATIENT METRICS
    # ----------------------------
    total_outpatient = outpatient_df.height

    outpatient_with_ham = outpatient_df.filter(pl.col("has_hamilton")).height

    pct_outpatient_with_ham = (
        outpatient_with_ham / total_outpatient * 100 if total_outpatient > 0 else 0.0
    )

    ham_outpatient = outpatient_df["n_hamilton"].sum()

    # ----------------------------
    # TOTAL HAMILTON
    # ----------------------------
    total_ham = pl.from_pandas(get_hamilton_scores()).height

    # ----------------------------
    # TABLE
    # ----------------------------
    overview = pl.DataFrame(
        {
            "Metric": [
                # Inpatient
                "Total admissions",
                "Admissions with Hamilton",
                "Percent admissions with Hamilton",
                "Hamilton scores during admissions",
                "Mean Hamilton scores per admission",
                # Outpatient
                "Total outpatient visits",
                "Outpatient visits with Hamilton",
                "Percent outpatient visits with Hamilton",
                "Hamilton scores in outpatient setting",
                # Overall
                "Total Hamilton scores (all)",
            ],
            "Value": [
                str(total_admissions),
                str(admissions_with_ham),
                f"{pct_admissions_with_ham:.1f}%",
                str(int(ham_in_admissions)),
                f"{mean_ham_per_admission:.2f}",
                str(total_outpatient),
                str(outpatient_with_ham),
                f"{pct_outpatient_with_ham:.1f}%",
                str(int(ham_outpatient)),
                str(total_ham),
            ],
        }
    )

    print("\n=== Hamilton Utilization Overview (F3 A-diagnosis cohort) ===")
    print(overview)


if __name__ == "__main__":
    inpatient_df = inpatient_pipeline().collect()
    outpatient_df = outpatient_pipeline().collect()

    result = print_overview_tables(inpatient_df, outpatient_df)
    print(result)
