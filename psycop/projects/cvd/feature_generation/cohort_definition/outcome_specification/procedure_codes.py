import polars as pl

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load

CVD_PROCEDURE_CODES = {
    "AMI": ["DI21", "DI22", "DI23"],
    "Stroke": ["DI6"],
    "PCI": ["KFNG02", "KFNG05", "KFNG96"],
    "CABG": ["KFNA", "KFNB", "KFNC", "KFND", "KFNE"],
    "Intracranial endovascular thrombolysis": ["KAAL10", "KAAL11"],
    "Other intracranial endovascular surgery": ["KAAL99"],
    "Iliac artery": ["KPDE", "KPDF", "KPDH", "KPDN", "KPDP", "KPDQ"],
    "Femoral artery": ["KPEE", "KPEF", "KPEH", "KPEN", "KPEP", "KPEQ"],
    "Popliteal artery and distal ": ["KPFE", "KPFG", "KPFH", "KPFN", "KPFP", "KPFQ"],
    "Amputation": ["KNFQ", "KNGQ", "KNHQ"],
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
