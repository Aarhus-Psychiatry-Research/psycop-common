import re
from typing import Literal, Optional, Union

import pandas as pd
from psycop_feature_generation.loaders.raw.load_text import load_text_split
from psycop_feature_generation.text_models.utils import stop_words


### standard preprocessing functions for text
def convert_series_to_lower_case(text_series: pd.Series) -> pd.Series:
    """Converts texts to lower case

    Args:
        text_series (pd.Series): Series containing texts

    Returns:
        pd.Series: Series containing only lower case
    """
    return text_series.str.lower()


def remove_symbols_from_series(text_series: pd.Series) -> pd.Series:
    """Removes symbols from texts

    Args:
        text_series (pd.Series): Series containing texts

    Returns:
        pd.Series: Series containing texts with symbols removed
    """
    res = []
    # text_series = text_series.replace()
    for row in text_series:
        text = re.sub("[^ÆØÅæøåA-Za-z0-9 ]+", "", row)
        res.append(text)

    return pd.Series(res)


def remove_stop_words_from_series(text_series: pd.Series) -> pd.Series:
    """Removes stop words from texts

    Args:
        text_series (pd.Series): Series containing texts

    Returns:
        pd.Series: Series containing texts with stop words removed
    """
    regex_stop_words = re.compile(r"\b%s\b" % r"\b|\b".join(map(re.escape, stop_words)))
    text_series = text_series.replace(regex_stop_words, value="", regex=True)

    return pd.Series(text_series)


### preprocessing for specific models for text
def text_preprocessing(
    text_sfi_names=Union[str, list[str]],
    include_sfi_name: bool = False,
    n_rows: Optional[int] = None,
    split_name=Literal["train", "val"],
) -> pd.DataFrame:
    """Preprocess texts by lower casing, removing

    Args:
        text_sfi_names (Union[str, list[str]]): Which sfi types to load. See `get_all_valid_text_sfi_names()` for valid sfi types.
        include_sfi_name (bool, optional): Whether to include the "overskrift" column, which includes sfi names. Defaults to False.
        n_rows (Optional[int], optional): Number of rows to include. If None, all rows are included. Defaults to None.
        split_name (_type_, optional): Splits to include. Defaults to Literal["train", "val"].

    Returns:
        pd.DataFrame: _description_
    """
    df = load_text_split(
        text_sfi_names=text_sfi_names,
        include_sfi_name=include_sfi_name,
        n_rows=n_rows,
        split_name=split_name,
    )

    df["value"] = convert_series_to_lower_case(df["value"])
    df["value"] = remove_symbols_from_series(df["value"])
    df["value"] = remove_stop_words_from_series(df["value"])

    return df
