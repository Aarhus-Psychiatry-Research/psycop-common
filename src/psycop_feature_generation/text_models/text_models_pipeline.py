"""Pipeline for fitting text models"""
import logging
import os.path
from collections.abc import Iterable
from datetime import datetime

from psycop_feature_generation.loaders.raw.sql_load import sql_load
from psycop_feature_generation.text_models.fit_text_models import (
    fit_bow,
    fit_lda,
    fit_tfidf,
)
from psycop_feature_generation.text_models.utils import save_text_model_to_dir

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.INFO)


def bow_model_pipeline(
    view: str = "psycop_train_val_all_sfis_all_years_lowercase_stopwords_and_symbols_removed",
    sfi_type: Iterable[str] = ["All_sfis"],
    n_rows: int = None,
    ngram_range: tuple = (1, 2),
    max_df: float = 0.95,
    min_df: int = 2,
    max_features: int = 500,
    save_path: str = "E:/shared_resources/text_models",
) -> str:
    """Pipeline for fitting and saving a bag-of-words model

    Args:
        view (str, optional): SQL table with text data to fit model on. Defaults to "psycop_train_val_all_sfis_all_years_lowercase_stopwords_and_symbols_removed".
        sfi_type (Iterable[str], optional): Which sfi types to include. Defaults to ["All_sfis"].
        n_rows (int, optional): How many rows to include in the loaded data. If None, all are included. Defaults to None.
        ngram_range (tuple, optional): The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted. All values of n such such that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams. Defaults to (1, 2).
        max_df (float, optional): The proportion of documents the words should appear in to be included. Defaults to 0.95.
        min_df (int, optional): Remove words occuring in less than min_df documents. Defaults to 2.
        max_features (int | None, optional): If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. Otherwise, all features are used. Defaults to 500.
        save_path (str, optional): Path where the model will be saved. Defaults to "E:/shared_resources/text_models".

    Returns:
        str: Log info on the path and filename of the fitted bow model.
    """
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
        f" {datetime.now().strftime('%H:%M:%S')}: Corpus loaded. Starting fitting bow model to corpus",
    )

    # fit model
    bow_vec, _ = fit_bow(
        corpus=corpus["text"].tolist(),
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
    )

    log.info(f" {datetime.now().strftime('%H:%M:%S')}: Bow model fitted")

    # save model to dir
    save_text_model_to_dir(model=bow_vec, save_path=save_path, filename=filename)

    return log.info(
        f" {datetime.now().strftime('%H:%M:%S')}: Bow model fit and saved at {save_path}/{filename}",
    )


def tfidf_model_pipeline(
    view: str = "psycop_train_val_all_sfis_all_years_lowercase_stopwords_and_symbols_removed",
    sfi_type: Iterable[str] = ["All_sfis"],
    n_rows: int = None,
    ngram_range: tuple = (1, 2),
    max_df: float = 0.95,
    min_df: int = 2,
    max_features: int = 500,
    save_path: str = "E:/shared_resources/text_models",
) -> str:
    """Pipeline for fitting and saving a term frequency-inverse document frequency (tf-idf) model

    Args:
        view (str, optional): SQL table with text data to fit model on. Defaults to "psycop_train_val_all_sfis_all_years_lowercase_stopwords_and_symbols_removed".
        sfi_type (Iterable[str], optional): Which sfi types to include. Defaults to ["All_sfis"].
        n_rows (int, optional): How many rows to include in the loaded data. If None, all are included. Defaults to None.
        ngram_range (tuple, optional): The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted. All values of n such such that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams. Defaults to (1, 2).
        max_df (float, optional): The proportion of documents the words should appear in to be included. Defaults to 0.95.
        min_df (int, optional): Remove words occuring in less than min_df documents. Defaults to 2.
        max_features (int | None, optional): If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. Otherwise, all features are used. Defaults to 500.
        save_path (str, optional): Path where the model will be saved. Defaults to "E:/shared_resources/text_models".

    Returns:
        str: Log info on the path and filename of the fitted tf-idf model.
    """

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

    corpus = sql_load(query=query, n_rows=n_rows)

    log.info(
        f" {datetime.now().strftime('%H:%M:%S')}: Corpus loaded. Starting fitting tfidf model to corpus",
    )

    # fit model
    tfidf_vec, _ = fit_tfidf(
        corpus=corpus["text"].tolist(),
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
    )

    log.info("Tfidf model fitted")

    # save model to dir
    save_text_model_to_dir(model=tfidf_vec, save_path=save_path, filename=filename)

    return log.info(
        f" {datetime.now().strftime('%H:%M:%S')}: Tfidf model fit and saved at {save_path}/{filename}",
    )


def lda_model_pipeline(
    view: str = "psycop_train_val_all_sfis_all_years_lowercase_stopwords_and_symbols_removed",
    sfi_type: Iterable[str] = ["All_sfis"],
    n_rows: int = None,
    ngram_range: tuple = (1, 2),
    max_df: float = 0.95,
    min_df: int = 2,
    max_features: int = 500,
    n_components: int = 20,
    n_top_words: int = 50,
    save_path: str = "E:/shared_resources/text_models",
) -> str:
    """Pipeline for fitting and saving a lda topic model

    Args:
        view (str, optional): SQL table with text data to fit model on. Defaults to "psycop_train_val_all_sfis_all_years_lowercase_stopwords_and_symbols_removed".
        sfi_type (Iterable[str], optional): Which sfi types to include. Defaults to ["All_sfis"].
        n_rows (int, optional): How many rows to include in the loaded data. If None, all are included. Defaults to None.
        ngram_range (tuple, optional): The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted. All values of n such such that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams. Defaults to (1, 2).
        max_df (float, optional): The proportion of documents the words should appear in to be included. Defaults to 0.95.
        min_df (int, optional): Remove words occuring in less than min_df documents. Defaults to 2.
        max_features (int | None, optional): If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. Otherwise, all features are used. Defaults to 500.
        save_path (str, optional): Path where the model will be saved. Defaults to "E:/shared_resources/text_models".
        n_components (int, optional): Number of components/topics to include. Defaults to 20.
        n_top_words (into, optional): Number of words to extract from each topic. Defaults to 50.

    Returns:
        str: Log info on the path and filename of the fitted bow model.
    """
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

    corpus = sql_load(query=query, n_rows=n_rows)

    log.info(
        f" {datetime.now().strftime('%H:%M:%S')}: Corpus loaded. Starting fitting lda model to corpus",
    )

    # fit model
    lda, model_topics = fit_lda(
        corpus=corpus["text"].tolist(),
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
        f" {datetime.now().strftime('%H:%M:%S')}: Lda model fit and model and model topics saved at {save_path}/{filename}",
    )
