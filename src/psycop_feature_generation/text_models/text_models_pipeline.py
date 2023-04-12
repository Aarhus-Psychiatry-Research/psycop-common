"""Pipeline for fitting text models"""
import os.path
from datetime import datetime
import logging

from psycop_feature_generation.loaders.raw.sql_load import sql_load
from psycop_feature_generation.text_models.fit_text_models import (
    fit_bow,
    fit_tfidf,
    fit_lda,
)
from psycop_feature_generation.text_models.utils import save_text_model_to_dir

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


def bow_model_pipeline(
    view: str = None,
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
    filename = f"bow_{view}_ngram_range_{ngram_range_str}_max_df_{max_df_str}_min_df_{min_df}_max_features_{max_features}.pkl"

    # if model already exists:
    if os.path.isfile("E:/shared_resources/text_models/" + filename):
        return log.warning(
            f"Bow model with the chosen params already exists in dir: E:/shared_resources/text_models/{filename}. Stopping.",
        )

    # load preprocessed data from sql
    corpus = sql_load(query=f"SELECT * FROM fct.{view}", n_rows=n_rows)

    # fit model
    bow = fit_bow(
        corpus["text"],
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
    )

    # save model to dir
    save_text_model_to_dir(model=bow, save_path=save_path, filename=filename)

    return f"Bow model fit and saved at {save_path}/{filename}"


def tfidf_model_pipeline(
    view: str = None,
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
    filename = f"tfidf_{view}_ngram_range_{ngram_range_str}_max_df_{max_df_str}_min_df_{min_df}_max_features_{max_features}.pkl"

    # if model already exists:
    if os.path.isfile("E:/shared_resources/text_models/" + filename):
        return log.warning(
            f"Tfidf model with the chosen params already exists in dir: E:/shared_resources/text_models/{filename}. Stopping.",
        )

    # load preprocessed data from sql
    corpus = sql_load(query=f"SELECT * FROM fct.{view}", n_rows=n_rows)

    # fit model
    tfidf = fit_tfidf(
        corpus["text"],
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
    )

    # save model to dir
    save_text_model_to_dir(model=tfidf, save_path=save_path, filename=filename)

    return f"Tfidf model fit and saved at {save_path}/{filename}"


def lda_model_pipeline(
    view: str = None,
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
    filename = f"lda_{view}_ngram_range_{ngram_range_str}_max_df_{max_df_str}_min_df_{min_df}_max_features_{max_features}"

    # if model already exists:
    if os.path.isfile("E:/shared_resources/text_models/" + filename + ".pkl"):
        return log.warning(
            f"Lda model with the chosen params already exists in dir: E:/shared_resources/text_models/{filename}. Stopping.",
        )

    # load preprocessed data from sql
    corpus = sql_load(query=f"SELECT * FROM fct.{view}", n_rows=n_rows)

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

    # save model to dir
    save_text_model_to_dir(model=lda, save_path=save_path, filename=(filename+".pkl"))
    
    # save model topics to dir
    model_topics.to_csv(save_path+"/topics_"+filename + ".csv", index=False)

    return f"Lda model fit and model and model topics saved at {save_path}/{filename}"


lda_model_pipeline(
    view="psycop_all_sfis_all_years_lowercase_stopwords_and_symbols_removed",
    n_rows=10000,
)
