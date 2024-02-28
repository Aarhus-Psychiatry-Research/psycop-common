from collections.abc import Iterable
from pathlib import Path

import polars as pl
import yaml
from sklearn.feature_extraction.text import CountVectorizer

from psycop.common.feature_generation.text_models.encode_text_as_tfidf_scores import (
    encode_tfidf_values_to_df,
)
from psycop.common.feature_generation.text_models.text_model_paths import PREPROCESSED_TEXT_DIR
from psycop.common.global_utils.paths import TEXT_EMBEDDINGS_DIR


def get_present_state_examination_keywords() -> Iterable[str]:
    # load yaml file
    keywords_file_path = Path(__file__).parent / "pse_keywords.yaml"
    with keywords_file_path.open(encoding="utf8") as f:
        keywords = yaml.full_load(f)
    # only keep the values and flatten the list
    flattened_keywords = [keyword for sublist in keywords.values() for keyword in sublist]
    return {keyword.lower() for keyword in flattened_keywords}


if __name__ == "__main__":
    vocab = get_present_state_examination_keywords()
    vec = CountVectorizer(vocabulary=vocab)

    corpus = pl.read_parquet(
        source=PREPROCESSED_TEXT_DIR / "psycop_train_val_test_all_sfis_preprocessed.parquet",
        low_memory=True,
    )

    counts = encode_tfidf_values_to_df(vec, corpus["value"].to_list())  # type: ignore

    corpus = corpus.drop(["value"])
    counts = pl.concat([corpus, counts], how="horizontal")
    counts.write_parquet(TEXT_EMBEDDINGS_DIR / "pse_keyword_counts_all_sfis.parquet")
