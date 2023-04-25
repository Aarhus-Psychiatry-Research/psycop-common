"""Load text data from sql warehouse."""
from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from multiprocessing import Pool

import pandas as pd

from psycop_feature_generation.loaders.raw.sql_load import sql_load
from psycop_feature_generation.utils import data_loaders


def get_valid_text_sfi_names() -> set[str]:
    """Returns a set of valid text sfi names. Notice that 'Konklusion' is replaced
    by 'Vurdering/konklusion' in 2020, so make sure to use both. 'Ordination'
    was replaced by 'Ordination, Psykiatry' in 2022, but 'Ordination,
    Psykiatri' is not included in the table. Use with caution.

    Returns:
        Set[str]: Set of valid text sfi names
    """
    return {
        "Observation af patient, Psykiatri",
        "Samtale med behandlingssigte",
        "Ordination",  # OBS replaced "Ordination, Psykiatri" in 01/02-22
        # but is not included in this table. Use with caution
        "Aktuelt psykisk",
        "Aktuelt socialt, Psykiatri",
        "Aftaler, Psykiatri",
        "Medicin",
        "Aktuelt somatisk, Psykiatri",
        "Objektivt psykisk",
        "Kontaktårsag",
        "Telefonkonsultation",
        "Journalnotat",
        "Telefonnotat",
        "Objektivt, somatisk",
        "Plan",
        "Semistruktureret diagnostisk interview",
        "Vurdering/konklusion",
    }


def _load_text_sfis_for_year(
    year: str,
    text_sfi_names: str | list[str],
    view: str | None = "FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret",
    n_rows: int | None = None,
) -> pd.DataFrame:
    """Loads clinical notes from sql from a specified year and matching
    specified text sfi names.

    Args:
        text_sfi_names (Union[str, list[str]]): Which types of notes to load.
        year (str): Which year to load
        view (str, optional): Which table to load.
            Defaults to "[FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret".
        n_rows (Optional[int], optional): Number of rows to load. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe with clinical notes
    """

    sql = (
        "SELECT dw_ek_borger, datotid_senest_aendret_i_sfien, fritekst"
        + f" FROM [fct].[{view}_{year}_inkl_2021_feb2022]"
        + f" WHERE overskrift IN {text_sfi_names}"
    )
    return sql_load(
        sql,
        database="USR_PS_FORSK",
        chunksize=None,
        n_rows=n_rows,
    )


def load_text_sfis(
    text_sfi_names: str | Iterable[str],
    n_rows: int | None = None,
) -> pd.DataFrame:
    """Loads all clinical notes that match the specified note from all years.

    Args:
        text_sfi_names (Union[str, list[str]]): Which note types to load. See
            `get_all_valid_text_sfi_names()` for valid note types.
        n_rows (Optional[int], optional): How many rows to load. Defaults to None.

    Raises:
        ValueError: If given invalid note type

    Returns:
        pd.DataFrame: Featurized clinical notes
    """
    if isinstance(text_sfi_names, str):
        text_sfi_names = [text_sfi_names]

    # check for invalid note types
    if not set(text_sfi_names).issubset(get_valid_text_sfi_names()):
        raise ValueError(
            "Invalid note type. Valid note types are: "
            + str(get_valid_text_sfi_names()),
        )

    # convert text_sfi_names to sql query
    text_sfi_names = "('" + "', '".join(text_sfi_names) + "')"

    view = "FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret"

    load_and_featurize = partial(
        _load_text_sfis_for_year,
        text_sfi_names=text_sfi_names,
        view=view,
        n_rows=n_rows,
    )

    years = list(range(2011, 2022))

    with Pool(processes=len(years)) as p:
        dfs = p.map(load_and_featurize, [str(y) for y in years])

    df = pd.concat(dfs)
    df = df.rename(
        {"datotid_senest_aendret_i_sfien": "timestamp", "fritekst": "value"},
        axis=1,
    )
    return df


@data_loaders.register("all_notes")
def load_all_notes(
    n_rows: int | None = None,
) -> pd.DataFrame:
    """Returns all notes from all years.

    Args:
        n_rows (Optional[int], optional): Number of rows to load. Defaults to None.

    Returns:
        pd.DataFrame: (Featurized) notes
    """
    return load_text_sfis(
        text_sfi_names=get_valid_text_sfi_names(),
        n_rows=n_rows,
    )


@data_loaders.register("aktuelt_psykisk")
def load_aktuel_psykisk(
    n_rows: int | None = None,
) -> pd.DataFrame:
    """Returns 'Aktuelt psykisk' notes from all years.

    Args:
        n_rows (Optional[int], optional): Number of rows to load. Defaults to None.

    Returns:
        pd.DataFrame: (Featurized) notes
    """
    return load_text_sfis(
        text_sfi_names="Aktuelt psykisk",
        n_rows=n_rows,
    )


@data_loaders.register("load_text_sfis")
def load_arbitrary_notes(
    text_sfi_names: str | list[str],
    n_rows: int | None = None,
) -> pd.DataFrame:
    """Returns one or multiple note types from all years.

    Args:
        text_sfi_names (Union[str, list[str]]): Which note types to load. See
            `get_all_valid_text_sfi_names()` for a list of valid note types.
        n_rows (Optional[int], optional): Number of rows to load. Defaults to None.

    Returns:
        pd.DataFrame: (Featurized) notes
    """
    return load_text_sfis(
        text_sfi_names,
        n_rows=n_rows,
    )
