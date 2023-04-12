"""Script for fitting text models"""

import pandas as pd
import pickle as pkl
from pathlib import Path
from typing import Any, List, Sequence
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from psycop_feature_generation.loaders.raw.load_text import load_all_notes


def fit_bow(
    corpus: Sequence[str],
    ngram_range: tuple = (1, 1),
    max_df: float = 0.95,
    min_df: int = 2,
    max_features: int = 500,
) -> CountVectorizer:
    """Fits a bag-of words model on a corpus

    Args:
        corpus (Sequence[str]): The corpus to fit on
        ngram_range (tuple, optional): The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted. All values of n such such that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams. Defaults to (1, 1).
        max_df (float, optional): The proportion of documents the words should appear in to be included. Defaults to 0.95.
        min_df (int, optional): Remove words occuring in less than min_df documents. Defaults to 2.
        max_features (int | None, optional): If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. Otherwise, all features are used. Defaults to 500.

    Returns:
        CountVectorizer: Fitted bow model
    """

    # Define vectorizer
    bow = CountVectorizer(
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
    )

    # Fit to corpus
    bow = bow.fit(corpus)

    return bow


def fit_tfidf(
    corpus: Sequence[str],
    ngram_range: tuple = (1, 1),
    max_df: float = 0.95,
    min_df: int = 2,
    max_features: int = 500,
) -> TfidfVectorizer:
    """Fits a term frequency-inverse document frequency (tf-idf) model on a corpus

    Args:
        corpus (Sequence[str]): The corpus to fit on
        stop_words (list[str] | None, optional): List containing stop words, all of which will be removed from the resulting tokens. Defaults to None.
        ngram_range (tuple, optional): The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted. All values of n such such that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams. Defaults to (1, 1).
        max_df (float, optional): The proportion of documents the words should appear in to be included. Defaults to 0.95.
        min_df (int, optional): Remove words occuring in less than min_df documents. Defaults to 2.
        max_features (int | None, optional): If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. Otherwise, all features are used. Defaults to 500.

    Returns:
        TfidfVectorizer: Fitted tfidf model
    """

    # Define vectorizer
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
    )

    # Fit to corpus
    tfidf = tfidf.fit(corpus)

    return tfidf


def fit_lda(
    corpus: Sequence[str],
    ngram_range: tuple = (1, 1),
    max_df: float = 0.95,
    min_df: int = 2,
    max_features: int = 500,
    n_components: int = 20,
    n_top_words: int = 10,
) -> LatentDirichletAllocation:
    """Fits a latent dirichlet allocation (LDA) topic model on text data.

    Args:
        corpus (Sequence[str]): The corpus to fit on
        ngram_range (tuple, optional): The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted. All values of n such such that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams. Defaults to (1, 1).
        max_df (float, optional): The proportion of documents the words should appear in to be included. Defaults to 0.95.
        min_df (int, optional): Remove words occuring in less than min_df documents. Defaults to 2.
        max_features (int | None, optional): If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. Otherwise, all features are used. Defaults to 500.
        n_components (int, optional): Number of topics/components. Defaults to 20.
        n_top_words (int | None, optional): The number of top words be extracted per topic. Only used when get_model_topics = True. Defaults to None.

    Returns:
        LatentDirichletAllocation: Fitted lda
        model_topics (pd.DataFrame): Dataframe with n_top_words of each topic
    """

    # Define vectorizer
    tf_vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
    )

    # Fit on corpus
    tf = tf_vectorizer.fit_transform(corpus)

    lda = LatentDirichletAllocation(
        n_components=n_components,
        random_state=1,
    ).fit(tf)

    # Get model topics
    model_topics = get_model_topics(lda, tf_vectorizer, n_top_words)

    return lda, model_topics


def get_model_topics(
    model: LatentDirichletAllocation, vectorizer: CountVectorizer, n_top_words: int = 10
) -> pd.DataFrame:
    """Get topics of LDA model

    Args:
        model (LatentDirichletAllocation): Fitted LDA model
        vectorizer (CountVectorizer): Fitted count vectorizer.
        n_top_words (int, optional): How many words of the top words to include in each topic. Defaults to 10.

    Returns:
        pd.DataFrame: Dataframe with n_top_words of each topic.
    """

    word_dict = {}
    topics = list(range(model.n_components))
    feature_names = vectorizer.get_feature_names_out()

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        word_dict[topics[topic_idx]] = top_features

    return pd.DataFrame(word_dict)


def save_text_model_to_dir(
    model: Any,
    filename: str,
):
    """
    Saves the model to a pickle file

    Args:
        model: The model to save
        filename: The filename to save the model to
    """
    filepath = Path("E:/") / "shared_resources" / "text_models" / filename

    print("Text model saved at this path:", filepath)

    with Path(filepath).open("wb") as f:
        pkl.dump(model, f)
