"""Load text data from sql warehouse."""

from __future__ import annotations

from typing import TYPE_CHECKING

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from psycop.common.model_training_v2.trainer.preprocessing.step import PresplitStep


import polars as pl


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
        "Ordination",
        "Ordination, Psykiatri",
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
        "Konklusion",
        "Misbrugsoplysninger",
        "Alkohol og rusmidler",
        "Skadelig brug/afhængighed af rusmidler",
        "Psykiatrikonference",
    }


def load_text_sfis(
    text_sfi_names: str | Iterable[str],
    include_sfi_name: bool = False,
    view: str | None = "Clozapin_fritekst_resultat",
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

    if isinstance(text_sfi_names, str):
        text_sfi_names = [text_sfi_names]

    # check for invalid note types
    if not set(text_sfi_names).issubset(get_valid_text_sfi_names()):
        raise ValueError(
            "Invalid note type. Valid note types are: " + str(get_valid_text_sfi_names())
        )

    # convert text_sfi_names to sql query
    text_sfi_names = "('" + "', '".join(text_sfi_names) + "')"

    sql = "SELECT dw_ek_borger, datotid_senest_aendret_i_sfien, fritekst"

    if include_sfi_name:
        sql += ", overskrift"
    view = "Clozapin_fritekst_resultat"

    sql += f" FROM [fct].[{view}]" + f" WHERE overskrift IN {text_sfi_names}"

    df = sql_load(sql, database="USR_PS_FORSK", n_rows=n_rows)

    df = df.rename({"datotid_senest_aendret_i_sfien": "timestamp", "fritekst": "value"}, axis=1)

    return df


def load_text_split(
    text_sfi_names: str | Iterable[str] | None,
    split_ids_presplit_step: PresplitStep,
    include_sfi_name: bool = False,
    n_rows: int | None = None,
) -> pd.DataFrame:
    """Loads specified text sfi and only keeps data from the specified split

    Args:
        text_sfi_names: Which sfi types to load. See `get_all_valid_text_sfi_names()` for valid sfi types.
        split_ids_presplit_step: PresplitStep that filters rows by split ids (e.g. RegionalFilter or FilterByOutcomeStratifiedSplits)
        include_sfi_name: Whether to include column with sfi name ("overskrift"). Defaults to False.
        n_rows: Number of rows to load. Defaults to None.

    Returns:
        pd.DataFrame: Chosen sfis from chosen splits
    """

    text_df = (
        load_text_sfis(
            text_sfi_names=text_sfi_names, include_sfi_name=include_sfi_name, n_rows=None
        )
        if text_sfi_names
        else load_all_notes(n_rows=None, include_sfi_name=include_sfi_name)
    )

    text_df = text_df.rename(
        {"datotid_senest_aendret_i_sfien": "timestamp", "fritekst": "value"}, axis=1
    )

    text_split_df = (
        split_ids_presplit_step.apply(pl.from_pandas(text_df).lazy()).collect().to_pandas()
    )
    # randomly sample instead of taking the first n_rows
    if n_rows is not None:
        text_split_df = text_split_df.sample(n=n_rows, replace=False)

    return text_split_df


def load_all_notes(n_rows: int | None = None, include_sfi_name: bool = False) -> pd.DataFrame:
    """Returns all notes from all years.

    Args:
        n_rows (Optional[int], optional): Number of rows to load. Defaults to None.
        include_sfi_name (bool, optional): Whether to include column with sfi name ("overskrift"). Defaults to False.

    Returns:
        pd.DataFrame: (Featurized) notes
    """
    return load_text_sfis(
        text_sfi_names=get_valid_text_sfi_names(), n_rows=n_rows, include_sfi_name=include_sfi_name
    )


def load_preprocessed_sfis(
    text_sfi_names: set[str] | None = None,
    corpus_name: str = "psycop_clozapine_train_val_all_sfis_preprocessed_v3",
) -> pd.DataFrame:
    """Returns preprocessed sfis from preprocessed view/SQL table that includes the "overskrift" column.
    Preprocessed views are created using the function text_preprocessing_pipeline under text_models/preprocessing.

    Args:
        text_sfi_names (str | list[str] | set[str] | None): Sfis to include.  Defaults to None, which includes all sfis.
        corpus_name (str, optional): Name of parquet with preprocessed sfis. Defaults to "psycop_clozapine_train_val_all_sfis_preprocessed".
        n_rows (int | None, optional): Number of rows to include. Defaults to None, which includes all rows.

    Returns:
        pd.DataFrame: Preprocessed sfis from preprocessed view/SQL table.
    """

    # load corpus
    # load corpus
    view = f"[{corpus_name}]"
    sql = f"SELECT * FROM [fct].{view}"

    if text_sfi_names:
        sfis_to_keep = ", ".join("?" for _ in text_sfi_names)

        sql += f" WHERE overskrift IN ({sfis_to_keep})"

    corpus = sql_load(query=sql, server="BI-DPA-PROD", database="USR_PS_Forsk", n_rows=None)

    return corpus
