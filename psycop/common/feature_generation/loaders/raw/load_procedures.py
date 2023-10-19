import pandas as pd
import polars as pl

from psycop.common.feature_generation.loaders.raw import sql_load


def load_heart_procedure_codes() -> pl.DataFrame:
    df = pl.from_pandas(
        sql_load(
            query="SELECT * FROM [fct].[FOR_hjerte_procedurekoder_inkl_2021_feb2022]"
        )
    ).rename({"datotid_udfoert": "timestamp", "procedurekode": "procedure_code"})
    return df


def finalize_feature_df(df: pl.DataFrame) -> pd.DataFrame:
    subset_df = df.select(["dw_ek_borger", "timestamp"]).with_columns(value=pl.lit(1))
    return subset_df.to_pandas()


def pci() -> pd.DataFrame:
    subset_df = load_heart_procedure_codes().filter(pl.col("pci") == 1)
    return finalize_feature_df(subset_df)


def cabg() -> pd.DataFrame:
    subset_df = load_heart_procedure_codes().filter(pl.col("cabg") == 1)
    return finalize_feature_df(subset_df)


def pad() -> pd.DataFrame:
    subset_df = (
        load_heart_procedure_codes()
        .with_columns(pl.sum(pl.col("^lower_limb_.*$")).alias("any_lower_limb"))
        .filter(pl.col("any_lower_limb") >= 1)
    )
    return finalize_feature_df(subset_df)


if __name__ == "__main__":
    df = pad()

    pass
