from collections.abc import Iterable
from pathlib import Path

import polars as pl
import yaml
from sklearn.feature_extraction.text import CountVectorizer

from psycop.common.feature_generation.loaders.raw.load_text import load_preprocessed_sfis
from psycop.common.feature_generation.text_models.encode_text_as_tfidf_scores import (
    encode_tfidf_values_to_df,
)
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

    # load preprocessed text from sql
    corpus = pl.from_pandas(load_preprocessed_sfis())

    counts = encode_tfidf_values_to_df(vec, corpus["value"].to_list())  # type: ignore

    corpus = corpus.drop(["value"])
    counts = pl.concat([corpus, counts], how="horizontal")
    counts.write_parquet(TEXT_EMBEDDINGS_DIR / "pse_keyword_counts_all_sfis.parquet")
