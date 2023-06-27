import polars as pl

from psycop.common.feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)


def add_time_from_first_visit(df: pl.DataFrame) -> pl.DataFrame:
    first_visit = (
        pl.from_pandas(
            physical_visits_to_psychiatry(
                n_rows=None,
                timestamps_only=False,
                return_value_as_visit_length_days=False,
                timestamp_for_output="start",
            ),
        )
        .groupby("dw_ek_borger")
        .agg(pl.col("timestamp").min().alias("first_visit"))
    )
    df = df.join(first_visit, on="dw_ek_borger", how="left")

    df = df.with_columns(
        (pl.col("timestamp") - pl.col("first_visit")).alias("time_from_first_visit"),
    )
    return df
