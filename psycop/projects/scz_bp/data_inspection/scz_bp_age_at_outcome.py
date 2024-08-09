"""Script to get an overview of the age at outcome for non-prevalent cases"""

# pyright: reportUnusedExpression=false

import plotnine as pn
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    RegionalFilter,  # type: ignore
)
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_eligible_config import (
    N_DAYS_WASHIN,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.first_scz_or_bp_diagnosis import (
    get_first_scz_or_bp_diagnosis_with_time_from_first_contact,
)


def first_scz_or_bp_after_washin() -> pl.DataFrame:
    first_scz_or_bp = get_first_scz_or_bp_diagnosis_with_time_from_first_contact()

    return first_scz_or_bp.filter(
        pl.col("time_from_first_contact") >= pl.duration(days=N_DAYS_WASHIN)
    ).select("dw_ek_borger", "timestamp", "source")


if __name__ == "__main__":
    first_diagnosis = first_scz_or_bp_after_washin()  #
    first_diagnosis = get_first_scz_or_bp_diagnosis_with_time_from_first_contact().select(
        "dw_ek_borger", "timestamp", "source"
    )
    birthday_df = pl.from_pandas(birthdays())

    age_df = first_diagnosis.join(birthday_df, on="dw_ek_borger", how="inner").with_columns(
        ((pl.col("timestamp") - pl.col("date_of_birth")).dt.days() / 365).alias("age")
    )

    train_val_df = (
        RegionalFilter(splits_to_keep=["train", "val"])
        .apply(age_df.lazy())
        .collect()
        .with_columns(split=pl.lit("train"))
    )
    test_df = (
        RegionalFilter(splits_to_keep=["test"])
        .apply(age_df.lazy())
        .collect()
        .with_columns(split=pl.lit("test"))
    )

    age_df = pl.concat([train_val_df, test_df], how="vertical")

    age_df.filter(pl.col("source") == "bp")["age"].describe()
    age_df.filter(pl.col("source") == "scz")["age"].describe()

    age_df.with_columns(
        pl.col("age")
        .cut(
            breaks=[0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 99],
            labels=[
                "0",
                "15",
                "16",
                "16",
                "17",
                "18",
                "19",
                "20",
                "21",
                "22",
                "23",
                "24",
                "25",
                "26",
                "27",
                "28",
                "29",
                "30",
                "99",
            ],
        )
        .cast(pl.Int16)
        .alias("age_bin")
    ).group_by("age_bin").count().sort(by="age_bin")

    age_df.group_by(pl.col("age").round(0)).count().sort("count")
    (
        pn.ggplot(age_df, pn.aes(x="age", fill="split"))
        + pn.geom_density(alpha=0.5)
        + pn.geom_vline(xintercept=18, linetype="dashed")
        + pn.facet_wrap("~source")
        + pn.theme_minimal()
        + pn.theme(legend_position="bottom")
    )
    age_df.to_pandas().groupby("split")["age"].describe()

    (
        pn.ggplot(age_df, pn.aes(x="age"))
        + pn.geom_histogram()
        + pn.geom_vline(xintercept=18, linetype="dashed")
        + pn.facet_wrap("~source")
    )

    (
        pn.ggplot(age_df, pn.aes(x="age", color="source"))
        + pn.stat_ecdf()
        + pn.geom_vline(xintercept=18, linetype="dashed")
        + pn.labs(title="Cumulative density")
    )

    (
        pn.ggplot(age_df, pn.aes(x="age"))
        + pn.geom_histogram()
        + pn.geom_vline(xintercept=15, linetype="dashed")
        + pn.geom_vline(xintercept=60, linetype="dashed")
        + pn.theme_minimal()
        + pn.labs(x="Age at diagnosis", y="Count")
    )
    age_df.groupby("source").count()
    age_df.filter(pl.col("age") < 18).groupby("source").count()
