"""Pipeline for fitting text models"""
import os.path
from datetime import datetime
import logging
from typing import Iterable
from psycop_feature_generation.loaders.raw.sql_load import sql_load
from psycop_feature_generation.text_models.fit_text_models import (
    fit_bow,
    fit_tfidf,
    fit_lda,
)
from psycop_feature_generation.text_models.utils import save_text_model_to_dir

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.INFO)


def bow_model_pipeline(
    view: str = None,
    sfi_type: Iterable[str] = ["All_sfis"],
    n_rows: int = None,
    ngram_range: tuple = (1, 1),
    max_df: float = 0.95,
    min_df: int = 2,
    max_features: int = 500,
    save_path: str = "E:/shared_resources/text_models",
) -> str:
    # create model filename from params
    max_df_str = str(max_df).replace(".", "")
    ngram_range_str = "".join(c for c in str(ngram_range) if c.isdigit())
    sfi_type_str = "".join(sfi_type).replace(" ", "")
    filename = f"bow_{view}_sfi_type_{sfi_type_str}_ngram_range_{ngram_range_str}_max_df_{max_df_str}_min_df_{min_df}_max_features_{max_features}.pkl"

    # if model already exists:
    if os.path.isfile("E:/shared_resources/text_models/" + filename):
        return log.warning(
            f"Bow model with the chosen params already exists in dir: E:/shared_resources/text_models/{filename}. Stopping.",
        )

    # load preprocessed data from sql
    log.info(f" {datetime.now().strftime('%H:%M:%S')}. Starting to load corpus")

    query = f"SELECT * FROM fct.{view}"

    if sfi_type != ["All_sfis"]:
        if len(sfi_type) == 1:
            query += f" WHERE overskrift in ('{sfi_type[0]}')"
        else:
            query += f" WHERE overskrift in ('{sfi_type[0]}'"
            for sfi in sfi_type[1:]:
                query += f",'{sfi}'"
            query += ")"

    corpus = sql_load(query=query, n_rows=n_rows)

    log.info(
        f" {datetime.now().strftime('%H:%M:%S')}: Corpus loaded. Starting fitting bow model to corpus"
    )

    # fit model
    bow = fit_bow(
        corpus["text"],
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
    )

    log.info(f" {datetime.now().strftime('%H:%M:%S')}: Bow model fitted")

    # save model to dir
    save_text_model_to_dir(model=bow, save_path=save_path, filename=filename)

    return log.info(
        f" {datetime.now().strftime('%H:%M:%S')}: Bow model fit and saved at {save_path}/{filename}"
    )


def tfidf_model_pipeline(
    view: str = None,
    sfi_type: Iterable[str] = ["All_sfis"],
    n_rows: int = None,
    ngram_range: tuple = (1, 1),
    max_df: float = 0.95,
    min_df: int = 2,
    max_features: int = 500,
    save_path: str = "E:/shared_resources/text_models",
) -> str:
    # create model filename from params
    max_df_str = str(max_df).replace(".", "")
    ngram_range_str = "".join(c for c in str(ngram_range) if c.isdigit())
    sfi_type_str = "".join(sfi_type).replace(" ", "")
    filename = f"tfidf_{view}_sfi_type_{sfi_type_str}_ngram_range_{ngram_range_str}_max_df_{max_df_str}_min_df_{min_df}_max_features_{max_features}.pkl"

    # if model already exists:
    if os.path.isfile("E:/shared_resources/text_models/" + filename):
        return log.warning(
            f"Tfidf model with the chosen params already exists in dir: E:/shared_resources/text_models/{filename}. Stopping.",
        )

    # load preprocessed data from sql
    log.info(f" {datetime.now().strftime('%H:%M:%S')}: Starting to load corpus")

    query = f"SELECT * FROM fct.{view}"

    if sfi_type != ["All_sfis"]:
        if len(sfi_type) == 1:
            query += f" WHERE overskrift in ('{sfi_type[0]}')"
        else:
            query += f" WHERE overskrift in ('{sfi_type[0]}'"
            for sfi in sfi_type[1:]:
                query += f",'{sfi}'"
            query += ")"

    corpus = sql_load(query=f"SELECT * FROM fct.{view}", n_rows=n_rows)

    log.info(
        f" {datetime.now().strftime('%H:%M:%S')}: Corpus loaded. Starting fitting tfidf model to corpus"
    )

    # fit model
    tfidf = fit_tfidf(
        corpus["text"],
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
    )

    log.info("Tfidf model fitted")

    # save model to dir
    save_text_model_to_dir(model=tfidf, save_path=save_path, filename=filename)

    return log.info(
        f" {datetime.now().strftime('%H:%M:%S')}: Tfidf model fit and saved at {save_path}/{filename}"
    )


def lda_model_pipeline(
    view: str = None,
    sfi_type: Iterable[str] = ["All_sfis"],
    n_rows: int = None,
    ngram_range: tuple = (1, 1),
    max_df: float = 0.95,
    min_df: int = 2,
    max_features: int = 500,
    n_components: int = 20,
    n_top_words: int = 10,
    save_path: str = "E:/shared_resources/text_models",
) -> str:
    # create model filename from params
    max_df_str = str(max_df).replace(".", "")
    ngram_range_str = "".join(c for c in str(ngram_range) if c.isdigit())
    sfi_type_str = "".join(sfi_type).replace(" ", "")
    filename = f"lda_{view}_sfi_type_{sfi_type_str}_ngram_range_{ngram_range_str}_max_df_{max_df_str}_min_df_{min_df}_max_features_{max_features}_n_components_{n_components}_n_top_words_{n_top_words}"

    # if model already exists:
    if os.path.isfile("E:/shared_resources/text_models/" + filename + ".pkl"):
        return log.warning(
            f"Lda model with the chosen params already exists in dir: E:/shared_resources/text_models/{filename}. Stopping.",
        )

    # load preprocessed data from sql
    log.info(f" {datetime.now().strftime('%H:%M:%S')}: Starting to load corpus")

    query = f"SELECT * FROM fct.{view}"

    if sfi_type != ["All_sfis"]:
        if len(sfi_type) == 1:
            query += f" WHERE overskrift in ('{sfi_type[0]}')"
        else:
            query += f" WHERE overskrift in ('{sfi_type[0]}'"
            for sfi in sfi_type[1:]:
                query += f",'{sfi}'"
            query += ")"

    corpus = sql_load(query=f"SELECT * FROM fct.{view}", n_rows=n_rows)

    log.info(
        f" {datetime.now().strftime('%H:%M:%S')}: Corpus loaded. Starting fitting lda model to corpus",
    )

    # fit model
    lda, model_topics = fit_lda(
        corpus["text"],
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        n_components=n_components,
        n_top_words=n_top_words,
    )

    log.info("Lda model fitted")

    # save model to dir
    save_text_model_to_dir(model=lda, save_path=save_path, filename=(filename + ".pkl"))

    # save model topics to dir
    model_topics.to_csv(save_path + "/topics_" + filename + ".csv", index=False)

    return log.info(
        f" {datetime.now().strftime('%H:%M:%S')}: Lda model fit and model and model topics saved at {save_path}/{filename}"
    )
