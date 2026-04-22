import polars as pl

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.projects.psychometrics.hamilton.cohort.outcome_specification.hamilton_score import (
    get_hamilton_scores,
)


def load_admissions_with_both_timestamps() -> pl.LazyFrame:
    """
    Returns a LazyFrame with the three columns:
    dw_ek_borger, datotid_start, datotid_slut.
    """
    view = "[psykometri_indlaeggelser]"
    sql = "SELECT * FROM [fct]." + view

    df = sql_load(sql)

    df = (
        pl.from_pandas(df)
        .select(
            [pl.col("dw_ek_borger"), pl.col("datotid_indlaeggelse"), pl.col("datotid_udskrivning")]
        )
        .lazy()
    )

    return df


def admissions_with_hamilton_counts() -> pl.LazyFrame:
    admissions = load_admissions_with_both_timestamps()

    hamilton = (
        pl.from_pandas(get_hamilton_scores())
        .lazy()
        .select([pl.col("dw_ek_borger"), pl.col("timestamp").alias("hamilton_timestamp")])
    )

    joined = (
        admissions.join(hamilton, left_on="dw_ek_borger", right_on="dw_ek_borger", how="left")
        .filter(
            (pl.col("hamilton_timestamp") >= pl.col("datotid_indlaeggelse"))
            & (pl.col("hamilton_timestamp") <= pl.col("datotid_udskrivning"))
        )
        .with_columns(
            (
                pl.col("dw_ek_borger").cast(pl.Utf8)
                + "_"
                + pl.col("datotid_indlaeggelse").dt.strftime("%Y%m%d%H%M%S")
            ).alias("admission_id")
        )
    )

    agg = (
        joined.group_by(
            ["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"]
        )
        .agg(pl.count("hamilton_timestamp").alias("n_hamilton_scores"))
        .with_columns((pl.col("n_hamilton_scores") > 0).alias("has_hamilton"))
    )

    all_admissions = admissions.with_columns(
        (
            pl.col("dw_ek_borger").cast(pl.Utf8)
            + "_"
            + pl.col("datotid_indlaeggelse").dt.strftime("%Y%m%d%H%M%S")
        ).alias("admission_id")
    ).select(["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"])

    result = (
        all_admissions.join(
            agg,
            on=["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"],
            how="left",
        )
        .with_columns(
            pl.col("n_hamilton_scores").fill_null(0), pl.col("has_hamilton").fill_null(False)
        )
        .sort("dw_ek_borger")
    )
    return result


def summary_per_patient(admissions_df: pl.DataFrame) -> pl.DataFrame:
    return (
        admissions_df.group_by("dw_ek_borger")
        .agg(
            [
                pl.count("admission_id").alias("total_admissions"),
                pl.sum("has_hamilton").alias("admissions_with_hamilton"),
            ]
        )
        .sort("dw_ek_borger")
    )


def print_overview_tables(admissions_hamilton: pl.DataFrame) -> None:
    """
    Prints an overview :

     - Total number of admissions
     - Number of admissions with at least one Hamilton score
     - Percentage of admissions with Hamilton
     - Number of Hamilton scores during admissions
     - Number of Hamilton scores in the outpatient context
     - Total number of Hamilton score observations
    """

    total_admissions = admissions_hamilton.height
    admissions_with_ham = admissions_hamilton.filter(pl.col("has_hamilton")).height
    pct_admissions_with_ham = (
        admissions_with_ham / total_admissions * 100 if total_admissions > 0 else 0.0
    )

    ham_in_admission = admissions_hamilton["n_hamilton_scores"].sum()

    total_ham = pl.from_pandas(get_hamilton_scores()).height

    ham_outpatient = total_ham - ham_in_admission

    overview = pl.DataFrame(
        {
            "Metric": [
                "Admissions total",
                "Admissions with Hamilton",
                "Percent admissions with Hamilton",
                "Hamilton scores during admissions",
                "Hamilton scores in outpatient context",
                "Total number of Hamilton scores",
            ],
            "Value": [
                str(total_admissions),
                str(admissions_with_ham),
                f"{pct_admissions_with_ham:.1f}' '%",
                str(int(ham_in_admission)),
                str(int(ham_outpatient)),
                str(total_ham),
            ],
        }
    )

    print("\n=== Overview table ===")
    print(overview)


if __name__ == "__main__":
    admissions_hamilton = admissions_with_hamilton_counts().collect()
    print("=== Admissions with Hamilton score count ===")

    print_overview_tables(admissions_hamilton)
