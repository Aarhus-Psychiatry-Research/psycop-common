import polars as pl

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.projects.psychometrics.loaders.visits import ambulatory_visits_psykometri_2025
from psycop.projects.psychometrics.mania_rating_scales.cohort.outcome_specification.mania_rating_score import (
    get_mania_rating_scores,
)


def load_admissions_with_timestamps() -> pl.LazyFrame:
    """Return a LazyFrame with columns: dw_ek_borger, datotid_indlaeggelse, datotid_udskrivning."""
    view = "[psykometri_indlaeggelser]"
    sql = f"SELECT * FROM [fct].{view}"
    df = sql_load(sql)
    return (
        pl.from_pandas(df)
        .select(["dw_ek_borger", "datotid_indlaeggelse", "datotid_udskrivning"])
        .lazy()
    )


def load_ambulatory_visits() -> pl.LazyFrame:
    """Return a LazyFrame with columns dw_ek_borger and amb_visit_timestamp."""
    df_pd = ambulatory_visits_psykometri_2025(timestamps_only=True, timestamp_for_output="end")
    return (
        pl.from_pandas(df_pd)
        .select([pl.col("dw_ek_borger"), pl.col("timestamp").alias("amb_visit_timestamp")])
        .lazy()
    )


def admissions_with_mania_counts() -> pl.LazyFrame:
    admissions = load_admissions_with_timestamps()
    mania = (
        pl.from_pandas(get_mania_rating_scores())
        .lazy()
        .select([pl.col("dw_ek_borger"), pl.col("timestamp").alias("mania_timestamp")])
    )
    joined = (
        admissions.join(mania, on="dw_ek_borger", how="left")
        .filter(
            (pl.col("mania_timestamp") >= pl.col("datotid_indlaeggelse"))
            & (pl.col("mania_timestamp") <= pl.col("datotid_udskrivning"))
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
        .agg(pl.count("mania_timestamp").alias("n_mania_ratings"))
        .with_columns((pl.col("n_mania_ratings") > 0).alias("has_mania"))
    )
    all_adm = admissions.with_columns(
        (
            pl.col("dw_ek_borger").cast(pl.Utf8)
            + "_"
            + pl.col("datotid_indlaeggelse").dt.strftime("%Y%m%d%H%M%S")
        ).alias("admission_id")
    )
    return (
        all_adm.join(
            agg,
            on=["dw_ek_borger", "admission_id", "datotid_indlaeggelse", "datotid_udskrivning"],
            how="left",
        )
        .with_columns(pl.col("n_mania_ratings").fill_null(0), pl.col("has_mania").fill_null(False))
        .sort("dw_ek_borger")
    )


def compare_mania_same_day() -> pl.DataFrame:
    """Count mania rating observations that occur on the same date as an ambulatory visit."""
    mania = (
        pl.from_pandas(get_mania_rating_scores())
        .lazy()
        .select([pl.col("dw_ek_borger"), pl.col("timestamp").alias("mania_timestamp")])
        .with_columns(pl.col("mania_timestamp").dt.date().alias("date"))
    )
    ambulatory = load_ambulatory_visits().with_columns(
        pl.col("amb_visit_timestamp").dt.date().alias("date")
    )
    same_day = (
        mania.join(ambulatory, on=["dw_ek_borger", "date"], how="inner")
        .group_by("dw_ek_borger")
        .agg(pl.count("mania_timestamp").alias("n_same_day_mania"))
        .sort("dw_ek_borger")
    )
    return same_day.collect()


def print_overview_tables(admissions_mania: pl.DataFrame, same_day: pl.DataFrame) -> None:
    total_admissions = admissions_mania.height
    admissions_with_mania = admissions_mania.filter(pl.col("has_mania")).height
    pct_with_mania = admissions_with_mania / total_admissions * 100 if total_admissions else 0.0
    mania_in_admission = admissions_mania["n_mania_ratings"].sum()
    total_mania = pl.from_pandas(get_mania_rating_scores()).height
    mania_same_day = same_day["n_same_day_mania"].sum()

    overview = pl.DataFrame(
        {
            "Metric": [
                "Total admissions",
                "Admissions with mania rating",
                "Percent admissions with mania rating",
                "Mania ratings during admissions",
                "Total number of mania ratings",
                "Mania ratings on same day as ambulatory visit",
            ],
            "Value": [
                str(total_admissions),
                str(admissions_with_mania),
                f"{pct_with_mania:.1f} %",
                str(int(mania_in_admission)),
                str(total_mania),
                str(int(mania_same_day)),
            ],
        }
    )
    print("\n=== Overview table ===")
    print(overview)


if __name__ == "__main__":
    admissions_mania = admissions_with_mania_counts().collect()
    same_day_counts = compare_mania_same_day()
    print_overview_tables(admissions_mania, same_day_counts)
