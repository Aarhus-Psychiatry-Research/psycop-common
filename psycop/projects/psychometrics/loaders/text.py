"""Load text data from sql warehouse."""

from __future__ import annotations

from typing import TYPE_CHECKING

from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.projects.psychometrics.text_models.text_sfi_list import (
    get_400_most_common_text_sfi_names,
    get_clinical_relevant_text_sfi_names,
)

if TYPE_CHECKING:
    import pandas as pd

    from psycop.common.model_training_v2.trainer.preprocessing.step import PresplitStep


import polars as pl


def load_text_sfis(
    text_sfi_names: str | None = None,
    include_sfi_name: bool = False,
    view: str = "psykometri_SFI_fritekst_resultater",
    n_rows: int | None = None,
) -> pd.DataFrame:
    """Loads clinical notes from sql from which only
    Args:
        text_sfi_names (str | None): Which SFI types to load. Use "clinical relveant" for clinically relevant sfi, "400 most common" for the 400 most common names, or None to load all names.
        include_sfi_name (bool): Whether to include the SFI name column ("overskrift"). Defaults to False.
        view (str, optional): Which SQL view to query. Defaults to "psykometri_SFI_fritekst_resultater".
        n_rows (int | None, optional): Maximum number of rows to retrieve. Defaults to None (all rows)."""

    if text_sfi_names == "clinical relevant":
        sfi_names = get_clinical_relevant_text_sfi_names()
    elif text_sfi_names == "400 most common":
        sfi_names = get_400_most_common_text_sfi_names()
    else:
        sfi_names = None

    sql = "SELECT dw_ek_borger, datotid_senest_aendret_i_sfien, fritekst"
    if include_sfi_name:
        sql += ", overskrift"

    sql += f" FROM [fct].[{view}]"
    if sfi_names is not None:
        sql_names = "('" + "', '".join(sfi_names) + "')"
        sql += f" WHERE overskrift IN {sql_names}"

    df = sql_load(sql, database="USR_PS_FORSK", n_rows=n_rows)
    df = df.rename({"datotid_senest_aendret_i_sfien": "timestamp", "fritekst": "value"}, axis=1)
    return df


def load_text_split(
    split_ids_presplit_step: PresplitStep,
    text_sfi_names: str | None = None,
    include_sfi_name: bool = False,
    n_rows: int | None = None,
) -> pd.DataFrame:
    """Load specified text SFI(s) and keep only rows belonging to the given split.

    Args:
        text_sfi_names (str | None): Which SFI types to load. Use "clinical relveant" for clinically relevant sfi, "400 most common" for the 400 most common names, or None to load all names.
        split_ids_presplit_step (PresplitStep): Step that filters rows by split IDs(e.g. Random_2025_split, RegionalFilter or FilterByOutcomeStratifiedSplits).
        include_sfi_name (bool): Include the SFI name column ("overskrift").
            Defaults to False.
        n_rows (int | None): Number of rows to return after split filtering.
            Defaults to None (return all rows).

    Returns:
        pd.DataFrame: Selected SFI notes from the chosen split.
    """
    # Load notes according to the requested SFI set
    if text_sfi_names is not None:
        text_df = load_text_sfis(
            text_sfi_names=text_sfi_names, include_sfi_name=include_sfi_name, n_rows=None
        )
    else:
        text_df = load_all_notes(n_rows=None, include_sfi_name=include_sfi_name)

    # Rename columns for downstream processing
    text_df = text_df.rename(
        {"datotid_senest_aendret_i_sfien": "timestamp", "fritekst": "value"}, axis=1
    )

    text_split_df = (
        split_ids_presplit_step.apply(pl.from_pandas(text_df).lazy()).collect().to_pandas()
    )

    # Optional random sampling
    if n_rows is not None:
        text_split_df = text_split_df.sample(n=n_rows, replace=False)

    return text_split_df


def load_all_notes(
    view: str | None = "psykometri_SFI_fritekst_resultater",
    n_rows: int | None = None,
    include_sfi_name: bool = False,
) -> pd.DataFrame:
    """Returns all notes regardless of sfi_type from all years.

    Args:
        view (str): Which sql table to load. Defaults to psykometri_SFI_fritekst_resultater.
        n_rows (Optional[int], optional): Number of rows to load. Defaults to None.
        include_sfi_name (bool, optional): Whether to include column with sfi name ("overskrift"). Defaults to False.

    Returns:
        pd.DataFrame: (Featurized) notes
    """
    sql = "SELECT dw_ek_borger, datotid_senest_aendret_i_sfien, fritekst"

    if include_sfi_name:
        sql += ", overskrift"
    view = "psykometri_SFI_fritekst_resultater"

    sql += f" FROM [fct].[{view}]"

    df = sql_load(sql, database="USR_PS_FORSK", n_rows=n_rows)

    df = df.rename({"datotid_senest_aendret_i_sfien": "timestamp", "fritekst": "value"}, axis=1)

    return df


def load_preprocessed_sfis(
    split_ids_presplit_step: PresplitStep,
    text_sfi_names: set[str] | None = None,
    corpus_name: str = "psycop_psykometri_train_val_test_all_sfis_preprocessed_added_psyk_konf",
) -> pd.DataFrame:
    """Returns preprocessed sfis from preprocessed view/SQL table that includes the "overskrift" column.
    Preprocessed views are created using the function text_preprocessing_pipeline under text_models/preprocessing.

    Args:
        text_sfi_names (str | list[str] | set[str] | None): Sfis to include.  Defaults to None, which includes all sfis.
        split_ids_presplit_step: PresplitStep that filters rows by split ids (e.g. RegionalFilter or FilterByOutcomeStratifiedSplits)
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

    corpus_split = (
        split_ids_presplit_step.apply(pl.from_pandas(corpus).lazy()).collect().to_pandas()
    )

    return corpus_split
