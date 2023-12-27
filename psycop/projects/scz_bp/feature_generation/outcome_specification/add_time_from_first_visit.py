import polars as pl

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load


def time_of_first_contact_to_psychiatry() -> pl.DataFrame:
    view = "FOR_kohorte_indhold_pt_journal_inkl_2021_feb2022"

    first_contact = (
        pl.from_pandas(  # type: ignore
            sql_load(  # type: ignore
                f"SELECT dw_ek_borger, datotid_start FROM [fct].[{view}]",
                format_timestamp_cols_to_datetime=True,  # type: ignore
                n_rows=None,  # type: ignore
            ),
        )
        .groupby("dw_ek_borger")
        .agg(pl.col("datotid_start").min().alias("first_contact"))
    )
    return first_contact


def add_time_from_first_contact_to_psychiatry(df: pl.DataFrame) -> pl.DataFrame:
    first_contact = time_of_first_contact_to_psychiatry()

    df = df.join(first_contact, on="dw_ek_borger", how="left")

    df = df.with_columns(
        (pl.col("timestamp") - pl.col("first_contact")).alias(
            "time_from_first_contact",
        ),
    )
    return df
