"""Pipeline for fitting text models"""
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Optional

from psycop_feature_generation.loaders.raw.sql_load import sql_load
from psycop_feature_generation.text_models.fit_text_models import fit_text_model
from psycop_feature_generation.text_models.utils import save_text_model_to_dir

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def text_model_pipeline(
    model: Literal["bow", "tfidf"],
    view: str = "psycop_train_val_all_sfis_all_years_lowercase_stopwords_and_symbols_removed",
    sfi_type: Sequence[str] = ("All_sfis",),
    n_rows: Optional[int] = None,
    ngram_range: tuple = (1, 2),
    max_df: float = 0.95,
    min_df: int = 2,
    max_features: int = 500,
    save_path: str = "E:/shared_resources/text_models",
) -> Any:
    """Pipeline for fitting and saving a bag-of-words or tfidf model

    Args:
        model (Literal[str]): Which model to use. Takes either "bow" or "tfidf".
        view (str, optional): SQL table with text data to fit model on. Defaults to "psycop_train_val_all_sfis_all_years_lowercase_stopwords_and_symbols_removed".
        sfi_type (Sequence[str], optional): Which sfi types to include. Defaults to ("All_sfis",).
        n_rows (int, optional): How many rows to include in the loaded data. If None, all are included. Defaults to None.
        ngram_range (tuple, optional): The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted. All values of n such such that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams. Defaults to (1, 2).
        max_df (float, optional): The proportion of documents the words should appear in to be included. Defaults to 0.95.
        min_df (int, optional): Remove words occuring in less than min_df documents. Defaults to 2.
        max_features (int | None, optional): If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. Otherwise, all features are used. Defaults to 500.
        save_path (str, optional): Path where the model will be saved. Defaults to "E:/shared_resources/text_models".

    Returns:
        str: Log info on the path and filename of the fitted text model.
    """
    # create model filename from params
    max_df_str = str(max_df).replace(".", "")
    ngram_range_str = "".join(c for c in str(ngram_range) if c.isdigit())
    sfi_type_str = "".join(sfi_type).replace(" ", "")
    filename = f"bow_{view}_sfi_type_{sfi_type_str}_ngram_range_{ngram_range_str}_max_df_{max_df_str}_min_df_{min_df}_max_features_{max_features}.pkl"

    # if model already exists:
    if Path("E:/shared_resources/text_models/" + filename).is_file():
        return log.warning(
            "Text model with the chosen params already exists in dir: E:/shared_resources/text_models/%s. Stopping.",
            filename,
        )

    query = f"SELECT * FROM fct.{view}"

    if sfi_type != ("All_sfis",):
        if len(sfi_type) == 1:
            query += f" WHERE overskrift in ('{sfi_type[0]}')"
        else:
            query += f" WHERE overskrift in ('{sfi_type[0]}'"
            for sfi in sfi_type[1:]:
                query += f",'{sfi}'"
            query += ")"

    corpus = sql_load(query=query, n_rows=n_rows)

    # fit model
    vec = fit_text_model(
        model=model,
        corpus=corpus["text"],
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
    )

    # save model to dir
    save_text_model_to_dir(model=vec, save_path=save_path, filename=filename)

    return None
