from collections.abc import Sequence
from typing import Optional

import pandas as pd
import polars as pl

from psycop.common.feature_generation.text_models.utils import stop_words
from psycop.common.global_utils.sql.writer import write_df_to_sql
from psycop.projects.forced_admission_inpatient_temp_val.feature_generation.modules.loaders.load_text_fa_2025 import (
    get_valid_text_sfi_names,
    load_text_sfis,
)


def text_preprocessing(df: pd.DataFrame, text_column_name: str = "value") -> pd.DataFrame:
    pl_df = pl.from_pandas(df)

    # Build regex
    regex_stop_words = "|".join([rf"\b{sw}\b" for sw in stop_words])
    regex_symbol_removal = r"[^ÆØÅæøåA-Za-z0-9 ]+"
    combined_regex = f"{regex_stop_words}|{regex_symbol_removal}"

    # Entire column processed in parallel by Polars
    pl_df = pl_df.with_columns(
        [pl.col(text_column_name).str.to_lowercase().str.replace_all(combined_regex, "")]
    )

    return pl_df.to_pandas()


def text_preprocessing_pipeline(sfi_type: Optional[Sequence[str] | str] = None) -> str:
    """Pipeline for preprocessing all sfis from given splits. Filtering of which sfis to include in features happens in the loader.

    Args:
        split_ids_presplit_step: PresplitStep that filters rows by split ids (e.g. RegionalFilter or FilterByOutcomeStratifiedSplits)
        n_rows (Optional[int], optional): How many rows to load. Defaults to None, which loads all rows.
        sfi_type (Optional[Sequence[str] | str], optional): Which sfi types to include. Defaults to None, which includes all sfis.

    Returns:
        str: Text describing where preprocessed text has been saved.
    """

    # Load text from splits
    df = load_text_sfis(text_sfi_names=sfi_type if sfi_type else get_valid_text_sfi_names())

    # preprocess
    df = text_preprocessing(df)

    sfis = "_".join(sfi_type) if sfi_type else "all_sfis"

    write_df_to_sql(
        df, f"psycop_forced_adm_temp_val_{sfis}_preprocessed_2020_2025", if_exists="replace"
    )

    return f"Text preprocessed and uploaded to SQL as _{sfis}_preprocessed"


if __name__ == "__main__":
    text_preprocessing_pipeline()
