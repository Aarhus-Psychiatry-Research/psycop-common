"""Loaders for structured SFI-data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.common.feature_generation.utils import data_loaders

if TYPE_CHECKING:
    import pandas as pd


def sfi_loader(
    aktivitetstypenavn: str | None = None,
    elementledetekst: str | None = None,
    n_rows: int | None = None,
    value_col: str = "numelementvaerdi",
) -> pd.DataFrame:
    """Load structured_sfi data. By default returns entire structured_sfi data
    view with numelementværdi as the value column.

    Args:
        aktivitetstypenavn (str): Fiter by type of structured_sfi, e.g. 'broeset_violence_checklist', 'selvmordsvurdering'. Defaults to None, in which case all sfis are loaded. # noqa: DAR102
        elementledetekst (str): elementledetekst to filter on, e.g. 'Sum', "Selvmordstanker". Defaults to None, in which case all sfis are loade.
        n_rows: Number of rows to return. Defaults to None which returns entire structured_sfi data view.
        value_col: Column to return as value col. Defaults to 'numelementvaerdi'.

    Returns:
        pd.DataFrame
    """
    view = "[FOR_SFI_uden_fritekst_resultater_psyk_somatik_inkl_2021_feb2022]"
    sql = f"SELECT dw_ek_borger, datotid_resultat_udfoert, {value_col} FROM [fct].{view} WHERE datotid_resultat_udfoert IS NOT NULL"

    if aktivitetstypenavn:
        sql += f" AND aktivitetstypenavn = '{aktivitetstypenavn}'"
    if elementledetekst:
        sql += f" AND elementledetekst = '{elementledetekst}'"

    df = sql_load(sql, database="USR_PS_FORSK", n_rows=n_rows)

    # Drop duplicate rows
    df = df.drop_duplicates(keep="first")

    # Drop rows with duplicate dw_ek_borger and datotid_resultat_udfoert
    # Data contained rows with scores reported at the same time for the same patient but with different values
    df = df.drop_duplicates(  # type: ignore
        subset=["datotid_resultat_udfoert", "dw_ek_borger"], keep="first"
    )

    df = df.rename(  # type: ignore
        columns={"datotid_resultat_udfoert": "timestamp", value_col: "value"}
    )

    return df.reset_index(drop=True)


@data_loaders.register("broeset_violence_checklist")
def broeset_violence_checklist(n_rows: int | None = None) -> pd.DataFrame:
    return sfi_loader(
        aktivitetstypenavn="Brøset Violence Checkliste (BVC)",
        elementledetekst="Sum",
        n_rows=n_rows,
        value_col="numelementvaerdi",
    )


@data_loaders.register("broeset_violence_checklist_physical_threats")
def broeset_violence_checklist_physical_threats(n_rows: int | None = None) -> pd.DataFrame:
    df = sfi_loader(
        aktivitetstypenavn="Brøset Violence Checkliste (BVC)",
        elementledetekst="Fysiske trusler",
        n_rows=n_rows,
        value_col="elementkode",
    )

    df["value"] = df["value"].replace(
        to_replace=["010BroesetNulPoint", "020BroesetEtPoint"], value=[0, 1], regex=False
    )

    return df


@data_loaders.register("selvmordsrisiko")
def selvmordsrisiko(n_rows: int | None = None) -> pd.DataFrame:
    df = sfi_loader(
        aktivitetstypenavn="Screening for selvmordsrisiko",
        elementledetekst="ScrSelvmordlRisikoniveauKonkl",
        n_rows=n_rows,
        value_col="elementkode",
    )

    df["value"] = df["value"].replace(
        to_replace=[
            "010ScrSelvmordKonklRisikoniveau1",
            "020ScrSelvmordKonklRisikoniveau2",
            "030ScrSelvmordKonklRisikoniveau3",
        ],
        value=[1, 2, 3],
        regex=False,
    )

    return df


@data_loaders.register("hamilton_d17")
def hamilton_d17(n_rows: int | None = None) -> pd.DataFrame:
    return sfi_loader(
        aktivitetstypenavn="Vurdering af depressionssværhedsgrad med HAM-D17",
        elementledetekst="Samlet score HAM-D17",
        n_rows=n_rows,
        value_col="numelementvaerdi",
    )


@data_loaders.register("mas_m")
def mas_m(n_rows: int | None = None) -> pd.DataFrame:
    return sfi_loader(
        aktivitetstypenavn="MAS-M maniscoringsskema (Modificeret Bech-Rafaelsen Maniskala)",
        elementledetekst="MAS-M score",
        n_rows=n_rows,
        value_col="numelementvaerdi",
    )


@data_loaders.register("height_in_cm")
def height_in_cm(n_rows: int | None = None) -> pd.DataFrame:
    return sfi_loader(
        aktivitetstypenavn="Måling af patienthøjde (cm)",
        elementledetekst="Højde i cm",
        n_rows=n_rows,
        value_col="numelementvaerdi",
    )


@data_loaders.register("weight_in_kg")
def weight_in_kg(n_rows: int | None = None) -> pd.DataFrame:
    df = sfi_loader(
        aktivitetstypenavn="Måling af patientvægt (kg)",
        elementledetekst="Vægt i kg",
        n_rows=n_rows,
        value_col="numelementvaerdi",
    )

    df = df[df["value"] > 0.5]

    return df


@data_loaders.register("bmi")
def bmi(n_rows: int | None = None) -> pd.DataFrame:
    df = sfi_loader(
        aktivitetstypenavn="Bestemmelse af Body Mass Index (BMI)",
        elementledetekst="BMI",
        n_rows=n_rows,
        value_col="numelementvaerdi",
    )

    df = df[(df["value"] > 10.0) & (df["value"] < 70.0)]

    return df


@data_loaders.register("no_temporary_leave")
def no_temporary_leave(n_rows: int | None = None) -> pd.DataFrame:
    df = sfi_loader(
        aktivitetstypenavn="Udgang, ordination",
        elementledetekst="Udgangstype",
        n_rows=n_rows,
        value_col="elementvaerdi",
    )

    df = df[df["value"] == "Ingen udgang"]
    df["value"] = 1

    return df


@data_loaders.register("temporary_leave")
def temporary_leave(n_rows: int | None = None) -> pd.DataFrame:
    df = sfi_loader(
        aktivitetstypenavn="Udgang, ordination",
        elementledetekst="Udgangstype",
        n_rows=n_rows,
        value_col="elementvaerdi",
    )

    df = df[
        (df["value"] == "Ledsaget udgang med følge af personale")
        | (df["value"] == "Ledsaget udgang med følge af pårørende")
        | (df["value"] == "Uledsaget udgang efter aftale med personale")
    ]
    df["value"] = 1

    return df


@data_loaders.register("supervised_temporary_leave")
def supervised_temporary_leave(n_rows: int | None = None) -> pd.DataFrame:
    df = sfi_loader(
        aktivitetstypenavn="Udgang, ordination",
        elementledetekst="Udgangstype",
        n_rows=n_rows,
        value_col="elementvaerdi",
    )

    df = df[
        (df["value"] == "Ledsaget udgang med følge af personale")
        | (df["value"] == "Ledsaget udgang med følge af pårørende")
    ]
    df["value"] = 1

    return df


@data_loaders.register("unsupervised_temporary_leave")
def unsupervised_temporary_leave(n_rows: int | None = None) -> pd.DataFrame:
    df = sfi_loader(
        aktivitetstypenavn="Udgang, ordination",
        elementledetekst="Udgangstype",
        n_rows=n_rows,
        value_col="elementvaerdi",
    )

    df = df[df["value"] == "Uledsaget udgang efter aftale med personale"]
    df["value"] = 1

    return df


def smoking_continuous() -> pd.DataFrame:
    """Gets smoking as a continuous variable. The unit is 'pack-years', i.e. number of years smoked times packs smoked per day."""
    df = pl.from_pandas(sql_load(query="SELECT * FROM [fct].[FOR_Rygning_SFI_inkl_2021_feb2022]"))

    df_pl_subset = df.select(
        ["dw_ek_borger", "datotid_senest_aendret_i_sfien", "numelementvaerdi"]
    ).filter(pl.col("numelementvaerdi").is_not_null())

    return df_pl_subset.rename(
        {"datotid_senest_aendret_i_sfien": "timestamp", "numelementvaerdi": "value"}
    ).to_pandas()


def smoking_categorical(mapping: dict[str, int] | None = None) -> pd.DataFrame:
    """Smoking as a categorical variable. See mapping within the function for definition, or provide your own."""
    if mapping is None:
        mapping = {
            "Ryger dagligt": 6,
            "Ryger": 5,
            "Ryger lejlighedsvis": 4,
            "Andet": 3,
            "Andet (f.eks. snus, e-cigaretter mv.)": 3,
            "Eks ryger": 2,
            "Tidligere ryger": 2,
            "Aldrig røget": 1,
        }

    df = pl.from_pandas(sql_load(query="SELECT * FROM [fct].[FOR_Rygning_SFI_inkl_2021_feb2022]"))

    df_pl_subset = df.select(
        ["dw_ek_borger", "datotid_senest_aendret_i_sfien", "rygning_samlet"]
    ).filter(pl.col("rygning_samlet").is_not_null())

    mapped = df_pl_subset.with_columns(
        pl.col("rygning_samlet").apply(lambda x: mapping.get(x), return_dtype=pl.Int16)  # type: ignore
    )

    return mapped.rename(
        {"rygning_samlet": "value", "datotid_senest_aendret_i_sfien": "timestamp"}
    ).to_pandas()


def _get_blood_pressure_pulse(
    subtype: Literal["Systolisk", "Diastolisk", "Pulsslag / min"],
) -> pl.LazyFrame:
    df = (
        pl.from_pandas(
            sql_load(query="SELECT * FROM [fct].[FOR_SFI_Blodtyk_Puls_psyk_somatik_inkl_2021]")
        )
        .lazy()
        .rename({"datotid_senest_aendret_i_sfien": "timestamp", "numelementvaerdi": "value"})
    )

    df.select(["dw_ek_borger", "timestamp", "value", "elementledetekst"])
    return df.filter(pl.col("elementledetekst") == pl.lit(subtype)).select(
        ["dw_ek_borger", "timestamp", "value"]
    )


def systolic_blood_pressure() -> pd.DataFrame:
    df = _get_blood_pressure_pulse(subtype="Systolisk")

    return df.collect().to_pandas()


def diastolic_blood_pressure() -> pd.DataFrame:
    df = _get_blood_pressure_pulse(subtype="Diastolisk")

    return df.collect().to_pandas()

def pulse() -> pd.DataFrame:
    df = _get_blood_pressure_pulse(subtype="Pulsslag / min")

    return df.collect().to_pandas()

