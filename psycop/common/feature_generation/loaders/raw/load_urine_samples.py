"""Loaders for medications."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load

log = logging.getLogger(__name__)


def urine_loader(n_rows: int | None = None) -> pd.DataFrame:
    """Load urine sample data.

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    cols = 'dw_ek_borger, datotidprv, prvnr, proeve_type, undersoegelse_tekst, proeve_type_tekst, materiale_tekst, bakterie_absolut_maengde, bakterienavn, bakterie_volumen'

    sql = f"SELECT {cols} FROM [fct].[FOR_Mikrobiologi_urin_inkl_2021_okt2024]"

    df = sql_load(sql, database="USR_PS_FORSK", n_rows=n_rows)

    return df


def _pathogen_group_a_list() -> list:
    return [
        "E. coli",
        "Enterokokker",
        "Klebsiella pneumoniae complex",
        "Enterobacter cloacae complex",
        "Citrobacter freundii complex",
        "Enterobacterales flere slags",
        "Acinetobacter baumannii complex",
        "Achromobacter xylosoxidans",
        "Raoutella planticola",
        "Enterokokker flere slags",
        "Raoutella ornithinolytica",
        "Hæmolytisk streptokokker",
        "Enterobacter asburiae",
        "Achromobacter xylosoxidans spp xylos.",
        "Actinotignum (tidl. Actinobaculum) schaalii",
        "Pseudomonas species  flere slags",
        "Aerococcus sanguinicola",
        "Streptococcus anginosus gruppen",
        "Acinetobacter baumannii",
        "Acinetobacter calcoaceticus",
        "Acinetobacter johnsonii",
        "Acinetobacter lwoffii",
        "Acinetobacter species",
        "Aerococcus urinae",
        "Aerococcus viridans",
        "Aeromonas species",
        "Alcaligenes faecalis",
        "Alcaligenes species",
        "Bacteroides fragilis",
        "Brevundimonas diminuta",
        "Burkholderia cepacia",
        "Burkholderia species",
        "Canidida albicans",
        "Citrobacter amalonaticus",
        "Citrobacter braakii",
        "Citrobacter farmeri",
        "Citrobacter freundii",
        "Citrobacter koseri",
        "Citrobacter sedlakii",
        "Citrobacter species",
        "Citrobacter youngae",
        "Comamonas testosteroni",
        "Delftia acidovorans",
        "Escherichia coli",
        "Enterobacter aerogenes",
        "Enterobacter asburiae (cloacae)",
        "Enterobacter cloacae",
        "Enterobacter gergoviae",
        "Enterobacter hormaechei (cloacae)",
        "Enterobacter kobei (cloacae)",
        "Enterobacter ludwigii",
        "Enterobacter sakazakii",
        "Enterobacter species",
        "Enterobacteriaceae",
        "Enterococcus avium",
        "Enterococcus casseliflavus",
        "Enterococcus durans",
        "Enterococcus faecalis",
        "Enterococcus faecium",
        "Enterococcus gallinarum",
        "Enterococcus hirae",
        "Enterococcus raffinosus",
        "Enterococcus species",
        "Escgerichia hermannii",
        "Escherichia species",
        "Fungi",
        "Haemophilus influenzae",
        "Haemophilus parainfluenzae",
        "Hafnia alvei",
        "Haemolytic streptococcus group A (S. pyogenes)",
        "Haemolytic streptococcus group B (S. agalactiae)",
        "Haemolytic streptococcus group C or G",
        "Haemolytic streptococcus group C (S. dysgalactiae)",
        "Haemolytic streptococcus group F (S. anginosus)",
        "Haemolytic streptococcus group G (S. dysgalactiae)",
        "Haemolytic streptococcus not group A, C or G",
        "Haemolytic streptococcus",
        "Klebsiella ornithinolytica",
        "Klebsiella oxytoca",
        "Klebsiella planticola",
        "Klebsiella pneumoniae",
        "Klebsiella species",
        "Kluyvera ascorbata",
        "Kluyvera cryocrescens",
        "Leclercia adecarboxylata",
        "Morganella morganii",
        "Morganella morganii, subsp.sibonii",
        "Morganella morganii,subsp. morganii",
        "Pantoea agglomerans",
        "Pantoea species",
        "Pasteurella multocida",
        "Proteus mirabilis",
        "Proteus penneri",
        "Proteus species",
        "Proteus vulgaris",
        "Providencia alcalifaciens",
        "Providencia rettgeri",
        "Providencia rustigianii",
        "Providencia species",
        "Pseudomonas aeruginosa",
        "Pseudomonas fluorescens",
        "Pseudomonas pseudoalcaligenes",
        "Pseudomonas putida",
        "Pseudomonas species",
        "Pseudomonas stutzeri",
        "Pseudomonas veronii",
        "Salmonella derby",
        "Salmonella dublin",
        "Salmonella heidelberg",
        "Salmonella species",
        "Salmonella typhimurium",
        "Serratia fonticola",
        "Serratia liquefaciens",
        "Serratia marcescens",
        "Serratia odorifera",
        "Serratia plymuthica",
        "Serratia species",
        "Shewanella putrefaciens",
        "Shigella boydii",
        "Sphinogmonas paucimobilis",
        "Staphylococcus aureus",
        "Staphylococcus lugdunensis",
        "Stenotrophomonas maltophilia",
        "Stenotrophomonas species",
        "Streptococcus pneumoniae",
        "Streptococcus salivarius",
    ]


def _pathogen_group_b_list() -> list:
    return [
        "Non-hæmolysende streptokokker",
        "Aerococcer",
        "Gærsvampe",
        "Non-hæmolysende strep. fl. slags",
        "Canadia species (non-albicans)",
        "Actinomyces neuii",
        "Aerococcus species",
        "Candadia dubliniensis",
        "Candadia glabrata",
        "Candadia kefyr",
        "Candadia krusei",
        "Candadia lusitaniae",
        "Candadia parapsilosis",
        "Candadia species",
        "Candadia tropicalis",
        "Corynebacterium jeikeium",
        "Cryptoococcus species",
        "Hafnia species",
        "Non-haemolytic streptococcus",
        "Shewanella species",
        "Staphylococcus capitis",
        "Staphylococcus caprae",
        "Staphylococcus epidermidis",
        "Staphylococcus haemolyticus",
        "Staphylococcus hominis",
        "Staphylococcus pasteuri",
        "Staphylococcus saprophyticus",
        "Staphylococcus simulans",
        "Staphylococcus warneri",
        "Streptococcus anginosus",
        "Streptococcus bovis/ gallolyticus",
        "Streptococcus constellatus",
        "Streptococcus lutetiensis",
        "Streptococcus mitis",
        "Streptococcus oralis",
    ]


def pathogen_group_a(n_rows: int | None = None) -> pd.DataFrame:
    df = urine_loader(n_rows=n_rows)

    df = df[df["bakterienavn"].isin(_pathogen_group_a_list())]

    df["gruppe"] = "A"

    return df


def pathogen_group_b(n_rows: int | None = None) -> pd.DataFrame:
    df = urine_loader(n_rows=n_rows)

    df = df[df["bakterienavn"].isin(_pathogen_group_b_list())]

    df["gruppe"] = "B"

    return df


def uvi_positive(n_rows: int | None = None) -> pd.DataFrame:
    df = pd.concat([pathogen_group_a(n_rows=n_rows), pathogen_group_b(n_rows=n_rows)]).reset_index(drop=True)

    df = df.sort_values(["dw_ek_borger", "prvnr", "datotidprv"])

    df["pureprv"] = df["prvnr"].duplicated(keep=False).astype(int)

    # Set default value
    df['value'] = 0

    # Create conditions for updating the 'value' column
    ## OBS - Keep bacteria Xs and make sure bac. A in condition 3 is the highest in the group
    condition_A_0 = (df['pureprv'] == 0) & (df['gruppe'] == "A") & (df['bakterie_absolut_maengde'] >= 10 ** 4)
    condition_B_0 = (df['pureprv'] == 0) & (df['gruppe'] == "B") & (df['bakterie_absolut_maengde'] >= 10 ** 5)
    condition_A_1 = (df['pureprv'] == 1) & (df['gruppe'] == "A") & (df['bakterie_absolut_maengde'] >= 10 ** 5)

    # Use numpy's where to set 'value' based on the conditions
    df['value'] = np.where(condition_A_0 | condition_B_0 | condition_A_1, 1, 0)

    # keep only positive samples
    df = df[df["value"] == 1]

    # keep only one row per prvnr
    df = df.drop_duplicates(subset=["dw_ek_borger", "prvnr"], keep="first")

    df = df.rename(columns={"datotidprv": "timestamp"})

    return df[["dw_ek_borger", "timestamp", "value"]]



if __name__ == "__main__":
    uvi_positive()
