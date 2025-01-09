"""Pipeline for fitting text models"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional

from psycop.common.feature_generation.loaders.raw.load_text import (
    load_preprocessed_sfis,
    load_text_split,
)
from psycop.common.feature_generation.text_models.fit_text_models import fit_text_model
from psycop.common.feature_generation.text_models.text_model_paths import TEXT_MODEL_DIR
from psycop.common.feature_generation.text_models.utils import save_text_model_to_shared_dir
from psycop.common.model_training_v2.trainer.preprocessing.step import PresplitStep
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    FilterByOutcomeStratifiedSplits,
)
from psycop.projects.clozapine.text_models.preprocessing import text_preprocessing

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
    split_ids_presplit_step: PresplitStep | None = None,
    corpus_name: str = "psycop_train_val_all_sfis_preprocessed",
    corpus_preproceseed: bool = False,
    sfi_type: Optional[Sequence[str] | str] = None,
    ngram_range: tuple[int, int] = (1, 1),
    max_df: float = 1.0,
    min_df: int = 1,
    max_features: int = 100,
    n_rows: int | None = None,
) -> Path:
    """Pipeline for fitting and saving a bag-of-words or tfidf model

    Args:
        model (Literal[str]): Which model to use. Takes either "bow" or "tfidf".
        corpus_name (str, optional): SQL table with text data (preprocessed or not) to fit model on. Defaults to "psycop_train_val_all_sfis_preprocessed".
        corpus_preproceseed (bool, optional): Whether the corpus is already preprocessed. Defaults to False.
        sfi_type (Sequence[str], optional): Which sfi types to include. Defaults to None.
        ngram_range (tuple, optional): The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted. All values of n such such that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams. Defaults to (1, 2).
        max_df (float, optional): The proportion of documents the words should appear in to be included. Defaults to 0.95.
        min_df (int, optional): Remove words occuring in less than min_df documents. Defaults to 2.
        max_features (int | None, optional): If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. Otherwise, all features are used. Defaults to 500.
        split_ids_presplit_step (PresplitStep, optional): A PresplitStep that filters rows by split ids. Defaults to None.
        n_rows (int | None, optional): Number of rows to load. Defaults to None, which loads all rows.

    Returns:
        str: Log info on the path and filename of the fitted text model.
    """

    split_ids_presplit_step = (
        split_ids_presplit_step
        if split_ids_presplit_step
        else FilterByOutcomeStratifiedSplits(splits_to_keep=["train", "val"])
    )

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

    if corpus_preproceseed:
        corpus = load_preprocessed_sfis(text_sfi_names=sfi_type, corpus_name=corpus_name)

    else:
        corpus = load_text_split(
            text_sfi_names=sfi_type,
            split_ids_presplit_step=split_ids_presplit_step,
            include_sfi_name=True,
            n_rows=n_rows,
        )

    corpus = text_preprocessing(corpus)

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
    save_text_model_to_shared_dir(model=vec, filename=filename)

    print(f"Text model fitted and saved as {model_path}")
    return model_path
