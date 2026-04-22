from collections.abc import Sequence
from typing import Literal, Optional

import pandas as pd
import polars as pl

from psycop.common.feature_generation.text_models.utils import stop_words
from psycop.common.global_utils.sql.writer import write_df_to_sql
from psycop.common.model_training_v2.trainer.preprocessing.step import PresplitStep
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    FilterByRandom2025Splits,
    RegionalFilter,
)
from psycop.projects.clozapine.loaders.text import get_valid_text_sfi_names, load_text_split


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


def text_preprocessing_pipeline(
    split_ids_presplit_step: PresplitStep,
    n_rows: Optional[int] = None,
    sfi_type: Optional[Sequence[str] | str] = None,
) -> str:
    """Pipeline for preprocessing all sfis from given splits. Filtering of which sfis to include in features happens in the loader.

    Args:
        split_ids_presplit_step: PresplitStep that filters rows by split ids (e.g. RegionalFilter or FilterByOutcomeStratifiedSplits)
        n_rows (Optional[int], optional): How many rows to load. Defaults to None, which loads all rows.
        sfi_type (Optional[Sequence[str] | str], optional): Which sfi types to include. Defaults to None, which includes all sfis.

    Returns:
        str: Text describing where preprocessed text has been saved.
    """

    splits_to_keep = (
        split_ids_presplit_step
        if split_ids_presplit_step
        else FilterByRandom2025Splits(splits_to_keep=["train", "val"])
    )

    # Load text from splits
    df = load_text_split(
        text_sfi_names=sfi_type if sfi_type else get_valid_text_sfi_names(),
        split_ids_presplit_step=split_ids_presplit_step,
        include_sfi_name=True,
        n_rows=n_rows,
    )

    # preprocess
    df = text_preprocessing(df)

    # save to parquet
    split_names = "_".join(splits_to_keep.splits_to_keep)  # type: ignore

    sfis = "_".join(sfi_type) if sfi_type else "all_sfis"

    write_df_to_sql(
        df,
        f"psycop_clozapine_{split_names}_{sfis}_preprocessed_added_psyk_konf_2025_random_split",
        if_exists="replace",
    )

    return f"Text preprocessed and uploaded to SQL as {split_names}_{sfis}_preprocessed"


if __name__ == "__main__":
    TRAIN_SPLITS: list[Literal["train", "val", "test"]] = ["train", "val", "test"]
    split_id_loaders = {
        "region": RegionalFilter(splits_to_keep=TRAIN_SPLITS),
        "id_outcome": FilterByRandom2025Splits(splits_to_keep=TRAIN_SPLITS),
    }
    SPLIT_TYPE = "id_outcome"

    text_preprocessing_pipeline(split_ids_presplit_step=split_id_loaders[SPLIT_TYPE], n_rows=None)
