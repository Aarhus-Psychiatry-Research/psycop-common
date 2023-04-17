import numpy as np
import pandas as pd
from psycop_feature_generation.text_models.fit_text_models import fit_text_model
from psycop_feature_generation.text_models.preprocessing import (
    convert_series_to_lower_case,
    remove_stop_words_from_series,
    remove_symbols_from_series,
)


def test_fit_bow_model():
    df = pd.DataFrame(
        {
            "dw_ek_borger": [1, 2, 3, 4],
            "timestamp": ["2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01"],
            "text": [
                "pt har ondt i maven pt",
                "udregning på torsdag",
                "pt fortæller om smerte i fod",
                "der er ingen der har spist morgenmad",
            ],
        },
    )

    df["text"] = convert_series_to_lower_case(df["text"])
    df["text"] = remove_symbols_from_series(df["text"])
    df["text"] = remove_stop_words_from_series(df["text"])

    bow = fit_text_model("bow", df["text"])
    transformed = bow.transform(df["text"])
    transformed = transformed.toarray()

    assert (np.array([[2], [0], [1], [0]]) == transformed).any()


def test_fit_tfidf_model():
    df = pd.DataFrame(
        {
            "dw_ek_borger": [1, 2, 3, 4],
            "timestamp": ["2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01"],
            "text": [
                "pt har ondt i maven pt",
                "udregning på torsdag",
                "pt fortæller om smerte i fod",
                "der er ingen der har spist morgenmad",
            ],
        },
    )

    df["text"] = convert_series_to_lower_case(df["text"])
    df["text"] = remove_symbols_from_series(df["text"])
    df["text"] = remove_stop_words_from_series(df["text"])

    tfidf = fit_text_model("tfidf", ["text"])
    transformed = tfidf.transform(df["text"])
    transformed = transformed.toarray()

    assert (np.array([[1], [0], [1], [0]]) == transformed).any()
