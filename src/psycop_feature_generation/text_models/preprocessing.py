import re
from typing import Literal

import pandas as pd
from psycop_feature_generation.text_models.data_handling import load_text_split
from psycop_feature_generation.text_models.utils import stop_words


### standard preprocessing functions for text
def convert_series_to_lower_case(text_series: pd.Series) -> pd.Series:
    return text_series.str.lower()


def remove_symbols_from_series(text_series=pd.Series) -> pd.Series:
    res = []
    for row in text_series:
        text = re.sub("[^ÆØÅæøåA-Za-z0-9 ]+", "", row)
        res.append(text)

    return pd.Series(res)


def remove_stop_words_from_series(text_series: pd.Series) -> pd.Series:
    regex_stop_words = re.compile(r"\b%s\b" % r"\b|\b".join(map(re.escape, stop_words)))
    res = []
    for row in text_series:
        text = re.sub(regex_stop_words, "", row)
        text = re.sub(" +", " ", text)
        res.append(text)

    return pd.Series(res)


### preprocessing for specific models for text
def text_preprocessing(
    text_sfi_names=str,
    include_sfi_name: bool = False,
    n_rows: int = None,
    split_name=Literal["train", "val"],
) -> pd.DataFrame:
    df = load_text_split(
        text_sfi_names=text_sfi_names,
        include_sfi_name=include_sfi_name,
        n_rows=n_rows,
        split_name=split_name,
    )

    df["text"] = convert_series_to_lower_case(df["text"])
    df["text"] = remove_symbols_from_series(df["text"])
    df["text"] = remove_stop_words_from_series(df["text"])

    return df
