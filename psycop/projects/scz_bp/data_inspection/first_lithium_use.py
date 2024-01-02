"""Script to investigate time of first lithium administration among bp patients"""

import plotnine as pn
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_medications import lithium
from psycop.projects.scz_bp.data_inspection.scz_bp_age_at_outcome import (
    first_scz_or_bp_after_washin,
)

# pyright: reportUnusedExpression=false


def get_bp_patients_in_training_data() -> pl.DataFrame:
    return first_scz_or_bp_after_washin().filter(pl.col("source") == "bp")


def get_first_administration_of_lithium() -> pl.DataFrame:
    return (
        pl.from_pandas(
            lithium(
                load_prescribed=False,
                load_administered=True,
            ),
        )
        .groupby("dw_ek_borger")
        .agg(pl.col("timestamp").min().alias("first_lithium"))
    )


if __name__ == "__main__":
    bp = get_bp_patients_in_training_data()
    first_lithium = get_first_administration_of_lithium()
    df = bp.join(first_lithium, on="dw_ek_borger", how="left")

    n_no_lithium = df["first_lithium"].null_count()

    # check time before the diagnosis lithium is administered, for those who
    # have it administered
    df = df.filter(pl.col("first_lithium").is_not_null()).with_columns(
        (pl.col("timestamp") - pl.col("first_lithium"))
        .dt.days()
        .alias("days_lithium_before_outcome"),
    )
    (
        pn.ggplot(df, pn.aes(x="days_lithium_before_outcome"))
        + pn.geom_histogram()
        + pn.annotate("text", x=-2000, y=25, label=" <-- diagnose før lithium")
        + pn.annotate("text", x=2000, y=25, label="lithium før diagnose -->")
    )
    pn.ggplot(
        df.filter(
            (pl.col("days_lithium_before_outcome") > -100)
            & (pl.col("days_lithium_before_outcome") < 100),
        ),
        pn.aes(x="days_lithium_before_outcome"),
    ) + pn.geom_histogram(bins=20)

    df["days_lithium_before_outcome"].describe()
    df.filter((pl.col("days_lithium_before_outcome") < 0) & (pl.col("days_lithium_before_outcome") > -90))

    # negativ = diagnose før lithium, positiv = lithum før diagnose
    # negativ = diagnose før lithium, positiv = lithum før diagnose
