import polars as pl

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load

CVD_PROCEDURE_CODES = {
    "MI": ["DI21", "DI22", "DI23"],
    "Stroke": ["DI6"],
    "PCI": ["KFNG02", "KFNG05", "KFNG96"],
    "CABG": ["KFNA", "KFNB", "KFNC", "KFND", "KFNE"],
    "Intracranial endovascular thrombolysis": ["KAAL10", "KAAL11"],
    "Other intracranial endovascular surgery": ["KAAL99"],
    "PAD": [
        "KPDE",
        "KPDF",
        "KPDH",
        "KPDN",
        "KPDP",
        "KPDQ",
        "KPEE",
        "KPEF",
        "KPEH",
        "KPEN",
        "KPEP",
        "KPEQ",
        "KPFE",
        "KPFG",
        "KPFH",
        "KPFN",
        "KPFP",
        "KPFQ",
        "KFNQ",  # Amputation
        "KNGQ",  # Amputation
        "KNHQ",  # Amputation
    ],
}


def get_cvd_procedures() -> pl.DataFrame:
    table = (
        pl.from_pandas(
            sql_load(query="SELECT * FROM [fct].[FOR_hjerte_procedurekoder_inkl_2021_feb2022]")
        )
        .select(["dw_ek_borger", "datotid_udfoert", "procedurekode"])
        .rename({"datotid_udfoert": "timestamp", "procedurekode": "procedure_code"})
    )
    return table
