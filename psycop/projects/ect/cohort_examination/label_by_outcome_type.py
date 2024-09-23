import polars as pl

from psycop.projects.ect.feature_generation.cohort_definition.outcome_specification.procedure_codes import (
    ECT_PROCEDURE_CODES,
)


def label_by_outcome_type(
    df: pl.DataFrame,
    procedure_col: str,
    output_col_name: str = "outcome_type",
    type2procedures: dict[str, list[str]] = ECT_PROCEDURE_CODES,
) -> pl.DataFrame:
    """Takes a dataframe with a column containing diagnosis/procedure codes adds a new column with the outcome type based on the diagnosis code."""

    # Initialise an empty column
    df = df.with_columns(pl.lit(None).alias(output_col_name))

    for label, procedures in reversed(type2procedures.items()):
        df = df.with_columns(
            pl.when(pl.col(procedure_col).is_in(procedures))
            .then(pl.lit(label))
            .otherwise(pl.col("outcome_type"))
            .alias(output_col_name)
        )

    return df
