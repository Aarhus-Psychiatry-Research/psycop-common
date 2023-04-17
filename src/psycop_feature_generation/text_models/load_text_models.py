"""Functions for loading fitted text models"""

import pickle as pkl
from pathlib import Path

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def _load_bow_model(
    filename: str,
    path_str: str = "E:/shared_resources/text_models",
) -> CountVectorizer:
    """Loads a bag-of-words model from a pickle file"""

    filepath = Path(path_str) / filename

    with Path(filepath).open("rb") as f:
        return pkl.load(f)


def _load_tfidf_model(
    filename: str,
    path_str: str = "E:/shared_resources/text_models",
) -> TfidfVectorizer:
    """Loads a tfidf model from a pickle file"""

    filepath = Path(path_str) / filename

    with Path(filepath).open("rb") as f:
        return pkl.load(f)


def _load_lda_model(
    filename: str,
    path_str: str = "E:/shared_resources/text_models",
) -> LatentDirichletAllocation:
    """Loads a lda model from a pickle file"""

    filepath = Path(path_str) / filename

    with Path(filepath).open("rb") as f:
        return pkl.load(f)
