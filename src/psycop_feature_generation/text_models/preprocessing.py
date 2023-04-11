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
    for word in text_series:
        text_series = re.sub("[^ÆØÅæøåA-Za-z0-9]+", " ", word)
        res.append(text_series)

    return pd.Series(res)


def remove_stop_words_from_series(text_series: pd.Series) -> pd.Series:
    regex_stop_words = re.compile(r"\b%s\b" % r"\b|\b".join(map(re.escape, stop_words)))
    res = []
    for word in text_series:
        text_series = re.sub(regex_stop_words, "", str(word))
        res.append(text_series)
    return pd.Series(res)


### preprocessing for specific models for text
def tfidf_preprocessing(
    text_sfi_names=str,
    split_name=Literal["train", "val"],
) -> pd.DataFrame:
    df_to_preprocess = load_text_split(text_sfi_names, split_name)

    df_to_preprocess["text"] = convert_series_to_lower_case(df_to_preprocess["text"])
    df_to_preprocess["text"] = remove_symbols_from_series(df_to_preprocess["text"])
    df_to_preprocess["text"] = remove_stop_words_from_series(df_to_preprocess["text"])
    preprocessed_df = df_to_preprocess

    return preprocessed_df
