"""Script to get an overview of the age at outcome for clozapine cases."""

from datetime import datetime

import polars as pl

from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    FilterByRandom2025Splits,
)
from psycop.projects.clozapine.feature_generation.cohort_definition.outcome_specification.combine_text_structured_clozapine_outcome import (
    combine_structured_and_text_outcome,
)
from psycop.projects.clozapine.loaders.demographics import birthdays


def get_clozapine_outcome() -> pl.DataFrame:
    return pl.from_pandas(combine_structured_and_text_outcome()).select(
        ["dw_ek_borger", "timestamp"]
    )


if __name__ == "__main__":
    outcome_df = get_clozapine_outcome()
    birthday_df = pl.from_pandas(birthdays())

    outcome_df_filter = outcome_df.filter(pl.col("timestamp") >= pl.lit(datetime(2014, 1, 1)))

    age_df = outcome_df_filter.join(birthday_df, on="dw_ek_borger", how="inner").with_columns(
        ((pl.col("timestamp") - pl.col("date_of_birth")).dt.total_days() / 365).alias("age")
    )

    train_val_df = (
        FilterByRandom2025Splits(splits_to_keep=["train", "val"])
        .apply(age_df.lazy())
        .collect()
        .with_columns(split=pl.lit("train_val"))
    )
    test_df = (
        FilterByRandom2025Splits(splits_to_keep=["test"])
        .apply(age_df.lazy())
        .collect()
        .with_columns(split=pl.lit("test"))
    )

    age_df = pl.concat([train_val_df, test_df], how="vertical")

    overall = age_df.select(
        pl.lit("overall").alias("split"),
        pl.col("age").mean().round(1).alias("mean_age"),
        pl.col("age").std().round(1).alias("sd_age"),
        pl.col("age").median().round(1).alias("median_age"),
        pl.col("age").quantile(0.25).round(1).alias("q1_age"),
        pl.col("age").quantile(0.75).round(1).alias("q3_age"),
    )

    summary = pl.concat(
        [
            age_df.group_by("split")
            .agg(
                pl.col("age").mean().round(1).alias("mean_age"),
                pl.col("age").std().round(1).alias("sd_age"),
                pl.col("age").median().round(1).alias("median_age"),
                pl.col("age").quantile(0.25).round(1).alias("q1_age"),
                pl.col("age").quantile(0.75).round(1).alias("q3_age"),
            )
            .sort("split"),
            overall,
        ],
        how="vertical",
    )

    print("\n=== Age at clozapine prescription ===")
    print(summary.to_pandas().to_string(index=False))

    print("\n=== Patient counts ===")
    print(
        age_df.group_by("split")
        .agg(pl.len().alias("n_patients"))
        .sort("split")
        .to_pandas()
        .to_string(index=False)
    )

    timestamps_pd = outcome_df.to_pandas()
