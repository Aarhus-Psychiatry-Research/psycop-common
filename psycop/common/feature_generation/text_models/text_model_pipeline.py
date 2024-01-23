"""Pipeline for fitting text models"""
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from psycop.common.feature_generation.text_models.fit_text_models import fit_text_model
from psycop.common.feature_generation.text_models.text_model_paths import (
    PREPROCESSED_TEXT_DIR,
    TEXT_MODEL_DIR,
)
from psycop.common.feature_generation.text_models.utils import save_text_model_to_dir

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def create_model_filename(
    model: Literal["bow", "tfidf"],
    corpus_name: str,
    ngram_range: tuple[int, int],
    max_df: float,
    min_df: int,
    max_features: Optional[int],
    sfi_type: Optional[Sequence[str]] = None,
) -> str:
    """Create model filename including all relevant informaiton about the model.

    Args:
        model (Literal[str]): Which model to use. Takes either "bow" or "tfidf".
        corpus_name (str): name of parquet with text data to fit model on.
        ngram_range (tuple): The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted.
        max_df (float): The proportion of documents the words should appear in to be included.
        min_df (int): Remove words occuring in less than min_df documents.
        max_features (int, optional): If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. Otherwise, all features are used.
        sfi_type (Sequence[str], optional): Which sfi types to include. Defauls to None, which includes all sfis.
    """
    max_df_str = str(max_df).replace(".", "")
    ngram_range_str = "".join(c for c in str(ngram_range) if c.isdigit())
    sfi_type_str = "all_sfis" if not sfi_type else "".join(sfi_type).replace(" ", "")

    return f"{model}_{corpus_name}_sfi_type_{sfi_type_str}_ngram_range_{ngram_range_str}_max_df_{max_df_str}_min_df_{min_df}_max_features_{max_features}.pkl"


def text_model_pipeline(
    model: Literal["bow", "tfidf"],
    corpus_name: str = "psycop_train_all_sfis_preprocessed",
    sfi_type: Optional[Sequence[str]] = None,
    ngram_range: tuple[int, int] = (1, 1),
    max_df: float = 1.0,
    min_df: int = 1,
    max_features: Optional[int] = 100,
) -> Path:
    """Pipeline for fitting and saving a bag-of-words or tfidf model

    Args:
        model (Literal[str]): Which model to use. Takes either "bow" or "tfidf".
        corpus_name (str, optional): SQL table with text data to fit model on. Defaults to "psycop_train_val_all_sfis_preprocessed".
        sfi_type (Sequence[str], optional): Which sfi types to include. Defaults to None.
        ngram_range (tuple, optional): The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted. All values of n such such that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams. Defaults to (1, 2).
        max_df (float, optional): The proportion of documents the words should appear in to be included. Defaults to 0.95.
        min_df (int, optional): Remove words occuring in less than min_df documents. Defaults to 2.
        max_features (int | None, optional): If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. Otherwise, all features are used. Defaults to 500.

    Returns:
        str: Log info on the path and filename of the fitted text model.
    """
    # create model filename from params
    filename = create_model_filename(
        model=model,
        corpus_name=corpus_name,
        sfi_type=sfi_type,
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
    )

    model_path = TEXT_MODEL_DIR / filename

    if model_path.exists():
        log.warning(f"Text model with the chosen params already exists in dir: {model_path}.")
        return model_path

    filter_list = [[("overskrift", "=", f"{sfi}")] for sfi in sfi_type] if sfi_type else None
    corpus = pd.read_parquet(
        path=PREPROCESSED_TEXT_DIR / f"{corpus_name}.parquet", filters=filter_list
    )

    # fit model
    vec = fit_text_model(
        model=model,
        corpus=corpus["value"],
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
    )

    # save model to dir
    save_text_model_to_dir(model=vec, filename=filename)

    print(f"Text model fitted and saved as {model_path}")
    return model_path
