"""Apply tfidf model to text data and save to disk"""
import glob
from pathlib import Path

import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

from psycop.common.feature_generation.text_models.encode_text_as_tfidf_scores import (
    encode_tfidf_values_to_df,
)
from psycop.common.feature_generation.text_models.text_model_paths import (
    PREPROCESSED_TEXT_DIR,
)
from psycop.common.feature_generation.text_models.utils import (
    chunk_dataframe,
    load_text_model,
)
from psycop.common.global_utils.paths import TEXT_EMBEDDINGS_DIR


def chunk_tfidf_chunking_process(
    tfidf_model: TfidfVectorizer,
    corpus: pl.DataFrame,
    embedding_dir: Path,
    model_name: str = "tfidf_model",
):
    #_remove_chunk_files_from_dir(embedding_dir)

    tfidf_df = _merge_tfidf_encoding_chunks(model_name, embedding_dir)

    chunked_corpus = chunk_dataframe(corpus, 50)

    for i, chunk in enumerate(chunked_corpus):
        tfidf_values = encode_tfidf_values_to_df(tfidf_model, chunk["value"].to_list())

        chunk = chunk.drop(columns=["value"])

        tfidf_notes = pl.concat([chunk, tfidf_values], how="horizontal")

        embedding_dir.mkdir(exist_ok=True, parents=True)
        tfidf_notes.write_parquet(
            embedding_dir / f"{model_name}_chunk_{i}.parquet",
        )

    tfidf_df = _merge_tfidf_encoding_chunks(model_name, embedding_dir)

    _remove_chunk_files_from_dir(embedding_dir)

    tfidf_df.write_parquet(
        embedding_dir / f"{model_name}.parquet",
    )


def _merge_tfidf_encoding_chunks(model_name: str, embedding_dir: Path) -> pl.DataFrame:
    dfs = _read_chunk_dfs_from_dir(
        model_name=model_name,
        embedding_dir=embedding_dir,
    )
    
    return _merge_dfs(dfs)


def _merge_dfs(dfs: list[pl.DataFrame]) -> pl.DataFrame:
    return pl.concat(dfs, how="vertical")


def _read_chunk_dfs_from_dir(
    model_name: str,
    embedding_dir: Path,
) -> list[pl.DataFrame]:
    df_dirs = glob.glob(str(embedding_dir / f"{model_name}_chunk_*.parquet"))

    return [pl.read_parquet(chunk_dir) for chunk_dir in df_dirs]


def _remove_chunk_files_from_dir(
    embedding_dir: Path,
):
    df_dirs = glob.glob(str(embedding_dir / "*_chunk_*.parquet"))

    for file in df_dirs:
        Path.unlink(Path(file))


if __name__ == "__main__":
    tfidf_model = load_text_model(
        "tfidf_psycop_train_all_sfis_preprocessed_sfi_type_all_sfis_ngram_range_12_max_df_099_min_df_2_max_features_10000.pkl",
    )

    corpus = pl.from_pandas(
        pd.read_parquet(
            path=PREPROCESSED_TEXT_DIR
            / "psycop_train_val_test_all_sfis_preprocessed.parquet",
        ),
    )

    model_name = "tfidf_train_val_test_all_sfis_ngram_range_12_max_df_099_min_df_2_max_features_10000"

    chunk_tfidf_chunking_process(
        tfidf_model=tfidf_model,  # type: ignore
        corpus=corpus,
        embedding_dir=TEXT_EMBEDDINGS_DIR,  # type: ignore
        model_name=model_name,
    )
