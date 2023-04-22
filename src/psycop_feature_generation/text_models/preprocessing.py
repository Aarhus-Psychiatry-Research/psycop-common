import re

import pandas as pd
from psycop_feature_generation.text_models.utils import stop_words


def text_preprocessing(
    df: pd.DataFrame,
    text_column_name: str = "value",
) -> pd.DataFrame:
    """Preprocess texts by lower casing, removing stopwords and symbols.

    Args:
        df (pd.DataFrame): Dataframe with a column containing text to clean.
        text_column_name (str): Name of column containing text. Defaults to "value".

    Returns:
        pd.DataFrame: _description_
    """

    regex_stop_words = re.compile(
        r"\b%s\b" % r"\b|\b".join(map(re.escape, stop_words)),
    )
    regex_symbols = r"[^ÆØÅæøåA-Za-z0-9 ]+"

    df[text_column_name] = (
        df[text_column_name]
        .str.lower()
        .replace(regex_stop_words, value="", regex=True)  # type: ignore
        .replace(regex_symbols, value="", regex=True)
        .replace(r"\s+", " ", regex=True)  # remove multiple whitespaces
    )

    return df
