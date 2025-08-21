"""Apply tfidf model to text data and save to disk"""

from collections.abc import Iterable
from time import time

import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

from psycop.common.feature_generation.loaders.raw.load_text import load_preprocessed_sfis
from psycop.common.feature_generation.text_models.utils import load_text_model
from psycop.common.global_utils.paths import TEXT_EMBEDDINGS_DIR


def encode_tfidf_values_to_df(model: TfidfVectorizer, text: Iterable[str]) -> pl.DataFrame:
    t0 = time()
    print("Start encoding")
    tfidf = model.transform(text)
    print(f"Encoding time: {time() - t0:.2f} seconds")
    return pl.DataFrame(tfidf.toarray(), schema=model.get_feature_names_out().tolist())  # type: ignore


if __name__ == "__main__":
    text_model_name = "tfidf_psycop_train_val_all_sfis_preprocessed_added_konklusion_sfi_type_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750"

    tfidf_model = load_text_model(f"{text_model_name}.pkl")

    # load preprocessed text from sql
    corpus = pl.from_pandas(load_preprocessed_sfis())

    print("Loaded text")
    tfidf_values = encode_tfidf_values_to_df(tfidf_model, corpus["value"].to_list())  # type: ignore

    corpus = corpus.drop(["value"])

    tfidf_notes = pl.concat([corpus, tfidf_values], how="horizontal")

    TEXT_EMBEDDINGS_DIR.mkdir(exist_ok=True, parents=True)
    tfidf_notes.write_parquet(
        TEXT_EMBEDDINGS_DIR
        / "text_tfidf_all_sfis_ngram_range_added_konklusion_12_max_df_09_min_df_2_max_features_750.parquet"
    )
