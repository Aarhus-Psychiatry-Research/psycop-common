import re
import sys
from collections.abc import Sequence
from typing import Literal, Optional

import pandas as pd

from psycop.common.feature_generation.text_models.utils import stop_words
from psycop.common.global_utils.sql.writer import write_df_to_sql
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    FilterByOutcomeStratifiedSplits,
    RegionalFilter,
)
from psycop.projects.clozapine.loaders.text import get_valid_text_sfi_names, load_text_split


def text_preprocessing(df: pd.DataFrame, text_column_name: str = "value") -> pd.DataFrame:
    """Preprocess texts by lower casing, removing stopwords and symbols.

    Args:
        df (pd.DataFrame): Dataframe with a column containing text to clean.
        text_column_name (str): Name of column containing text. Defaults to "value".

    Returns:
        pd.DataFrame: df with preprocessed text
    """
    # Define regex for stop words with empty string in beginning and end
    regex_stop_words_surrounded_by_empty_strings = [rf"\b{stop_word}\b" for stop_word in stop_words]
    regex_stop_words = "|".join(regex_stop_words_surrounded_by_empty_strings)

    # Define regex that removes symbols (by keeping everything else)
    regex_symbol_removal = r"[^ÆØÅæøåA-Za-z0-9 ]+"

    # combine
    regex_symbol_removal_and_stop_words = re.compile(f"{regex_stop_words}|{regex_symbol_removal}")

    # lower case and remove stop words and symbols
    total_rows = len(df)
    for i, text in enumerate(df[text_column_name]):
        df.at[i, text_column_name] = re.sub(regex_symbol_removal_and_stop_words, "", text.lower())

        # Update progress bar
        progress = (i + 1) / total_rows
        sys.stdout.write(f"\rProcessing: [{'#' * int(progress * 50):<50}] {progress * 100:.1f}%")
        sys.stdout.flush()

    print()  # Add a newline at the end of the progress bar

    return df


def text_preprocessing_pipeline(
    splits_to_keep: Sequence[Literal["train", "val", "test"]],
    n_rows: Optional[int] = None,
    sfi_type: Optional[Sequence[str] | str] = None,
) -> str:
    """Pipeline for preprocessing all sfis from given splits. Filtering of which sfis to include in features happens in the loader.

    Args:
        splits_to_keep:: Which splits to keep (train, val, test)
        n_rows (Optional[int], optional): How many rows to load. Defaults to None, which loads all rows.
        sfi_type (Optional[Sequence[str] | str], optional): Which sfi types to include. Defaults to None, which includes all sfis.

    Returns:
        str: Text describing where preprocessed text has been saved.
    """

    splits_to_keep = (
        splits_to_keep
        if splits_to_keep
        else FilterByOutcomeStratifiedSplits(splits_to_keep=["train", "val"])
    )

    # Load text from splits
    df = load_text_split(
        text_sfi_names=sfi_type if sfi_type else get_valid_text_sfi_names(),
        splits_to_keep=splits_to_keep,
        include_sfi_name=True,
        n_rows=n_rows,
    )

    # preprocess
    df = text_preprocessing(df)

    # save to parquet
    split_names = "_".join(splits_to_keep)  # type: ignore

    sfis = "_".join(sfi_type) if sfi_type else "all_sfis"

    write_df_to_sql(df, f"psycop_clozapine_{split_names}_{sfis}_preprocessed")

    return f"Text preprocessed and uploaded to SQL as {split_names}_{sfis}_preprocessed"


if __name__ == "__main__":
    TRAIN_SPLITS: list[Literal["train", "val", "test"]] = ["train", "val"]
    split_id_loaders = {
        "region": RegionalFilter(splits_to_keep=TRAIN_SPLITS),
        "id_outcome": FilterByOutcomeStratifiedSplits(splits_to_keep=TRAIN_SPLITS),
    }
    SPLIT_TYPE = "id_outcome"

    text_preprocessing_pipeline(split_ids_presplit_step=split_id_loaders[SPLIT_TYPE], n_rows=None)
