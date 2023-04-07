"""Functions for loading fitted text models"""

import pickle as pkl
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def _load_bow_model(
    filename: str, path: Path = Path("E:/") / "shared_resources" / "text_models"
) -> CountVectorizer:
    """Loads a bag-of-words model from a pickle file"""

    filepath = path / filename

    with Path(filepath).open("rb") as f:
        return pkl.load(f)


def _load_tfidf_model(
    filename: str, path: Path = Path("E:/") / "shared_resources" / "text_models"
) -> TfidfVectorizer:
    """Loads a tfidf model from a pickle file"""

    filepath = path / filename

    with Path(filepath).open("rb") as f:
        return pkl.load(f)


def _load_lda_model(
    filename: str, path: Path = Path("E:/") / "shared_resources" / "text_models"
) -> LatentDirichletAllocation:
    """Loads a lda model from a pickle file"""

    filepath = path / filename

    with Path(filepath).open("rb") as f:
        return pkl.load(f)
