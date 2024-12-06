import re

import pandas as pd

from psycop.common.feature_generation.text_models.utils import stop_words


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
    df[text_column_name] = (
        df[text_column_name]
        .str.lower()
        .replace(regex_symbol_removal_and_stop_words, value="", regex=True)  # type: ignore
    )

    return df
