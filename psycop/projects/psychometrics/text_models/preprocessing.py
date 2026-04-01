from typing import Optional

import pandas as pd
import polars as pl

from psycop.common.feature_generation.text_models.utils import stop_words
from psycop.common.global_utils.sql.writer import write_df_to_sql
from psycop.projects.psychometrics.loaders.text import load_all_notes


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


def text_preprocessing_pipeline(n_rows: Optional[int] = None) -> str:
    """Pipeline for preprocessing all text regardless of sfi_type"

    Args:
        n_rows (Optional[int], optional): How many rows to load. Defaults to None, which loads all rows.

    Returns:
        str: Text describing where preprocessed text has been saved.
    """

    # Load text from splits
    df = load_all_notes(include_sfi_name=True, n_rows=n_rows)

    # preprocess
    df = text_preprocessing(df)

    write_df_to_sql(df, "psycop_psychometrics_all_text_preprocessed", if_exists="replace")

    return "Text preprocessed and uploaded to SQL"


if __name__ == "__main__":
    text_preprocessing_pipeline(n_rows=1000)
