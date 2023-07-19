"""Load text data from sql warehouse."""
from __future__ import annotations

from functools import partial
from multiprocessing import Pool
from typing import TYPE_CHECKING, Literal

import pandas as pd

from psycop.common.feature_generation.application_modules.save_dataset_to_disk import (
    filter_by_split_ids,
    get_split_id_df,
)
from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.common.feature_generation.utils import data_loaders

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


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
    include_sfi_name: bool = False,
    view: str | None = "FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret",
    n_rows: int | None = None,
) -> pd.DataFrame:
    """Loads clinical notes from sql from a specified year and matching
    specified text sfi names.

    Args:
        year (str): Which year to load
        text_sfi_names (Union[str, list[str]]): Which types of notes to load.
        include_sfi_name (bool): Whether to include column with sfi name ("overskrift"). Defaults to False.
        view (str, optional): Which table to load.
            Defaults to "[FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret".
        n_rows (Optional[int], optional): Number of rows to load. Defaults to None.


    Returns:
        pd.DataFrame: Dataframe with clinical notes
    """

    sql = "SELECT dw_ek_borger, datotid_senest_aendret_i_sfien, fritekst"

    if include_sfi_name:
        sql += ", overskrift"

    sql += (
        f" FROM [fct].[{view}_{year}_inkl_2021_feb2022]"
        + f" WHERE overskrift IN {text_sfi_names}"
    )

    return sql_load(
        sql,
        database="USR_PS_FORSK",
        n_rows=n_rows,
    )


def load_text_sfis(
    text_sfi_names: str | Iterable[str],
    include_sfi_name: bool = False,
    n_rows: int | None = None,
) -> pd.DataFrame:
    """Loads all clinical notes that match the specified note from all years.

    Args:
        text_sfi_names (Union[str, list[str]]): Which sfi types to load. See
            `get_all_valid_text_sfi_names()` for valid sfi types.
        include_sfi_name (bool): Whether to include column with sfi name ("overskrift"). Defaults to False.
        n_rows (Optional[int], optional): Number of rows to load. Defaults to None.

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

    text_sfi_year_loader = partial(
        _load_text_sfis_for_year,
        text_sfi_names=text_sfi_names,
        include_sfi_name=include_sfi_name,
        view=view,
        n_rows=n_rows,
    )

    years = list(range(2011, 2022))

    with Pool(processes=len(years)) as p:
        dfs = p.map(text_sfi_year_loader, [str(y) for y in years])

    df = pd.concat(dfs)
    df = df.rename(
        {"datotid_senest_aendret_i_sfien": "timestamp", "fritekst": "value"},
        axis=1,
    )
    return df


def load_text_split(
    text_sfi_names: str | Iterable[str],
    split_name: Sequence[Literal["train", "val"]],
    include_sfi_name: bool = False,
    n_rows: int | None = None,
) -> pd.DataFrame:
    """Loads specified text sfi and only keeps data from the specified split

    Args:
        text_sfi_names (Union[str, list[str]]): Which sfi types to load. See `get_all_valid_text_sfi_names()` for valid sfi types.
        split_name (Literal["train", "val"], optional): Which splis to include. Defaults to Literal["train", "val"].
        include_sfi_name (bool, optional): Whether to include column with sfi name ("overskrift"). Defaults to False.
        n_rows (Optional[int, None], optional): Number of rows to load. Defaults to None.

    Returns:
        pd.DataFrame: Chosen sfis from chosen splits
    """

    text_df = load_text_sfis(
        text_sfi_names=text_sfi_names,
        include_sfi_name=include_sfi_name,
        n_rows=n_rows,
    )

    split_id_df = pd.concat(
        [get_split_id_df(split_name=split) for split in split_name],
    )

    text_split_df = filter_by_split_ids(
        df_to_split=text_df,
        split_id_df=split_id_df,
        split_name=split_name,  # type: ignore
    )

    return text_split_df


@data_loaders.register("all_notes")
def load_all_notes(
    n_rows: int | None = None,
    include_sfi_name: bool = False,
) -> pd.DataFrame:
    """Returns all notes from all years.

    Args:
        n_rows (Optional[int], optional): Number of rows to load. Defaults to None.
        include_sfi_name (bool, optional): Whether to include column with sfi name ("overskrift"). Defaults to False.

    Returns:
        pd.DataFrame: (Featurized) notes
    """
    return load_text_sfis(
        text_sfi_names=get_valid_text_sfi_names(),
        n_rows=n_rows,
        include_sfi_name=include_sfi_name,
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


@data_loaders.register("preprocessed_sfis")
def load_preprocessed_sfis(
    text_sfi_names: set[str] | None = None,
    corpus_name: str = "psycop.train_val_all_sfis_preprocessed",
) -> pd.DataFrame:
    """Returns preprocessed sfis from preprocessed view/SQL table that includes the "overskrift" column.
    Preprocessed views are created using the function text_preprocessing_pipeline under text_models/preprocessing.

    Args:
        text_sfi_names (str | list[str] | set[str] | None): Sfis to include.  Defaults to None, which includes all sfis.
        corpus_name (str, optional): Name of parquet with preprocessed sfis. Defaults to "psycop.train_val_all_sfis_preprocessed".
        n_rows (int | None, optional): Number of rows to include. Defaults to None, which includes all rows.

    Returns:
        pd.DataFrame: Preprocessed sfis from preprocessed view/SQL table.
    """

    # load corpus
    # if not text_sfi_names, include all sfis
    if not text_sfi_names:
        corpus = pd.read_parquet(
            path=f"E:/shared_resources/preprocessed_text/{corpus_name}.parquet",
        )
    # if text_sfi_names, include only chosen sfis
    else:
        filter_list = [[("overskrift", "=", f"{sfi}")] for sfi in text_sfi_names]
        corpus = pd.read_parquet(
            path=f"E:/shared_resources/preprocessed_text/{corpus_name}.parquet",
            filters=filter_list,
        )

    return corpus
