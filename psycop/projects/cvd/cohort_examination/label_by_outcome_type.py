import polars as pl

from psycop.projects.cvd.feature_generation.cohort_definition.outcome_specification.procedure_codes import (
    CVD_PROCEDURE_CODES,
)


def label_by_outcome_type(df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    # Reverse to get the most severe outcome first
    # Initialise an empty column
    df = df.with_columns(pl.lit(None).alias("outcome_type"))
    for outcome, substrings in reversed(CVD_PROCEDURE_CODES.items()):
        for substring in substrings:
            df = df.with_columns(
                pl.when(pl.col(group_col).str.contains(substring))
                .then(pl.lit(outcome))
                .otherwise("outcome_type")
                .alias("outcome_type")
            )

    return df
