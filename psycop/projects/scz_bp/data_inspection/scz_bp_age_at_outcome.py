"""Script to get an overview of the age at outcome for non-prevalent cases"""

# pyright: reportUnusedExpression=false

import plotnine as pn
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    _get_regional_split_df,  # type: ignore
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
    first_diagnosis = first_scz_or_bp_after_washin()
    birthday_df = pl.from_pandas(birthdays())

    train_val_ids = (
        _get_regional_split_df().filter(pl.col("region").is_in(["øst", "vest"])).collect()
    )

    age_df = (
        first_diagnosis.join(birthday_df, on="dw_ek_borger", how="inner")
        .filter(pl.col("dw_ek_borger").is_in(train_val_ids.get_column("dw_ek_borger")))
        .with_columns(
            ((pl.col("timestamp") - pl.col("date_of_birth")).dt.days() / 365).alias("age")
        )
    )

    age_df.filter(pl.col("source") == "bp")["age"].describe()
    age_df.filter(pl.col("source") == "scz")["age"].describe()

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

    age_df.groupby("source").count()
    age_df.filter(pl.col("age") < 18).groupby("source").count()
