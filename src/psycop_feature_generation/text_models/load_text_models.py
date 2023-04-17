"""Functions for loading fitted text models"""

import pickle as pkl
from pathlib import Path
from typing import Union

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def _load_text_model(
    filename: str,
    path_str: str = "E:/shared_resources/text_models",
) -> Union[CountVectorizer, TfidfVectorizer]:
    """Loads a text model from a pickle file"""

    filepath = Path(path_str) / filename

    with Path(filepath).open("rb") as f:
        return pkl.load(f)
