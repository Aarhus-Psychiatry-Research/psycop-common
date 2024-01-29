import numpy as np
import pandas as pd

from psycop.common.feature_generation.text_models.fit_text_models import fit_text_model
from psycop.common.feature_generation.text_models.preprocessing import text_preprocessing


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
        }
    )

    df = text_preprocessing(df, text_column_name="text")

    bow = fit_text_model("bow", df["text"], min_df=0, max_df=1, ngram_range=(1, 1), max_features=1)
    transformed = bow.transform(df["text"])
    transformed = transformed.toarray()  # type: ignore

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
        }
    )

    df = text_preprocessing(df, text_column_name="text")

    tfidf = fit_text_model(
        "tfidf", df["text"], min_df=0, max_df=1, ngram_range=(1, 1), max_features=1
    )
    transformed = tfidf.transform(df["text"])
    transformed = transformed.toarray()  # type: ignore

    assert (np.array([[0], [0], [0], [0]]) == transformed).any()
