import polars as pl

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load


def get_cvd_procedures() -> pl.DataFrame:
    table = (
        pl.from_pandas(
            sql_load(
                query="SELECT * FROM [fct].[FOR_hjerte_procedurekoder_inkl_2021_feb2022]",
            ),
        )
        .select(["dw_ek_borger", "datotid_udfoert", "procedurekode"])
        .rename({"datotid_udfoert": "timestamp", "procedurekode": "procedure_code"})
    )
    return table
