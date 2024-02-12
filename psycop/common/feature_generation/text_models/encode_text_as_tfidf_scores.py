"""Apply tfidf model to text data and save to disk"""
from collections.abc import Iterable
from time import time

import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

from psycop.common.feature_generation.text_models.text_model_paths import PREPROCESSED_TEXT_DIR
from psycop.common.feature_generation.text_models.utils import load_text_model
from psycop.common.global_utils.paths import TEXT_EMBEDDINGS_DIR


def encode_tfidf_values_to_df(model: TfidfVectorizer, text: Iterable[str]) -> pl.DataFrame:
    t0 = time()
    print("Start encoding")
    tfidf = model.transform(text)
    print(f"Encoding time: {time() - t0:.2f} seconds")
    return pl.DataFrame(tfidf.toarray(), schema=model.get_feature_names_out().tolist())  # type: ignore


if __name__ == "__main__":
    tfidf_model = load_text_model(
        "tfidf_psycop_train_all_sfis_preprocessed_sfi_type_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750.pkl"
    )

    corpus = pl.from_pandas(
        pd.read_parquet(path=PREPROCESSED_TEXT_DIR / "psycop_train_all_sfis_preprocessed.parquet")
    )

    tfidf_values = encode_tfidf_values_to_df(tfidf_model, corpus["value"].to_list())  # type: ignore

    corpus = corpus.drop(columns=["value"])

    tfidf_notes = pl.concat([corpus, tfidf_values], how="horizontal")

    TEXT_EMBEDDINGS_DIR.mkdir(exist_ok=True, parents=True)
    tfidf_notes.write_parquet(
        TEXT_EMBEDDINGS_DIR
        / "text_tfidf_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750.parquet"
    )
