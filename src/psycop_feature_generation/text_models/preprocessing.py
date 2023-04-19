import re
from typing import Literal, Optional, Union

import pandas as pd
from psycop_feature_generation.loaders.raw.load_text import load_text_split
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
        r"[^ÆØÅæøåA-Za-z0-9 ]+ \b%s\b" % r"\b|\b".join(map(re.escape, stop_words))
    )

    df["value"] = (
        df["value"]
        .replace(regex_stop_words, value="", regex=True)  # type: ignore
        .str.lower()
    )

    return df
