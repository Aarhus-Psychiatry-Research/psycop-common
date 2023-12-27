"""Script to get an overview of prevalent cases and how long of a wash-in to use"""
import plotnine as pn
import polars as pl

from psycop.common.model_training_v2.trainer.data.data_filters.geographical_split.make_geographical_split import (
    get_regional_split_df,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.first_scz_or_bp_diagnosis import (
    get_first_scz_or_bp_diagnosis_with_time_from_first_contact,
)


def print_summary_stats(df: pl.DataFrame) -> None:
    print("All patients:")
    print(df["days"].describe())
    print("Minimum 90 days from time of first contact:")
    print(df.filter(pl.col("days") > 90)["days"].describe())
    print("Minimum 180 days from time of first contact:")
    print(df.filter(pl.col("days") > 180)["days"].describe())


if __name__ == "__main__":
    df = get_first_scz_or_bp_diagnosis_with_time_from_first_contact()
    train_val_ids = (
        get_regional_split_df()
        .filter(pl.col("region").is_in(["øst", "vest"]))
        .collect()
    )
    # had subtracted one day from day of first diagnosis to avoid leakage
    df = df.filter(
        pl.col("dw_ek_borger").is_in(train_val_ids.get_column("dw_ek_borger"))
    ).with_columns(
        (pl.col("time_from_first_contact") + pl.duration(days=1))
        .dt.days()
        .alias("days")
    )

    less_than_zero = df.filter(pl.col("days") < 0)
    # 387 who received their diagnosis at a somatic contact, prior to their
    # first psychiatric contact. Setting their days from first contact to
    # diagnosis to 0
    df = df.with_columns(
        pl.when(pl.col("days") < 0).then(0).otherwise(pl.col("days")).alias("days")
    )

    pn.ggplot(df, pn.aes(x="days")) + pn.geom_histogram()

    scz_df = df.filter(pl.col("source") == "scz")
    bp_df = df.filter(pl.col("source") == "bp")

    print("Statistics for schizophrenia")
    print_summary_stats(scz_df)
    print("Statistics for bipolar disorder")
    print_summary_stats(bp_df)
