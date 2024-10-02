import polars as pl


def label_by_outcome_type(
    df: pl.DataFrame,
    procedure_col: str,
    type2procedures: dict[str, list[str]],
    output_col_name: str = "outcome_type",
) -> pl.DataFrame:
    """Takes a dataframe with a column containing diagnosis/procedure codes adds a new column with the outcome type based on the diagnosis code."""

    # Initialise an empty column
    df = df.with_columns(pl.lit(None).alias("outcome_type"))

    for label, procedures in reversed(type2procedures.items()):
        for substring in procedures:
            df = df.with_columns(
                pl.when(pl.col(procedure_col).str.contains(substring))
                .then(pl.lit(label))
                .otherwise("outcome_type")
                .alias(output_col_name)
            )

    return df
