import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

from psycop.projects.scz_bp.model_training.estimator_steps.cleanlab_processing import (
    CleanlabProcessing,
)


def sample_data() -> tuple[pd.DataFrame, pd.Series]:  # type: ignore
    X, y = make_classification(
        n_classes=2, weights=(0.9, 0.1), flip_y=0, n_features=5, n_samples=100, random_state=42
    )
    return pd.DataFrame(X), pd.Series(y, name="target")


@pytest.fixture
def clean_sample_data() -> tuple[pd.DataFrame, pd.Series]:  # type: ignore
    return sample_data()


@pytest.fixture
def noisy_sample_data() -> tuple[pd.DataFrame, pd.Series]:  # type: ignore
    X, y = sample_data()
    # flip random values of y
    y = y.apply(lambda x: 1 - x if np.random.rand() < 0.2 else x)

    return X, y


def test_keep_clean_data(clean_sample_data: tuple[pd.DataFrame, pd.Series]):  # type: ignore
    X, y = clean_sample_data
    cleanlab = CleanlabProcessing()
    X_res, y_res = cleanlab.fit_resample(X, y)  # type: ignore
    # tolerate some false positives
    assert len(X_res) <= len(X) * 1.02 and len(X_res) >= len(X) * 0.98  # noqa: PT018
    assert len(y_res) <= len(y) * 1.02 and len(y_res) >= len(y) * 0.98  # noqa: PT018


def test_remove_noisy_data(noisy_sample_data: tuple[pd.DataFrame, pd.Series]):  # type: ignore
    X, y = noisy_sample_data
    cleanlab = CleanlabProcessing()
    X_res, y_res = cleanlab.fit_resample(X, y)  # type: ignore
    assert len(X_res) < len(X) * 0.9
    assert len(y_res) < len(y) * 0.9
    assert len(X_res) + cleanlab.n_label_issues == len(X)
    assert len(y_res) + cleanlab.n_label_issues == len(y)


def test_cleanlab_processing_in_pipeline(
    noisy_sample_data: tuple[pd.DataFrame, pd.Series],  # type: ignore
    clean_sample_data: tuple[pd.DataFrame, pd.Series],  # type: ignore
):
    X_noisy, y_noisy = noisy_sample_data
    X_clean, y_clean = clean_sample_data

    pipe_with_cleanlab = Pipeline(
        [("cleanlab_processing", CleanlabProcessing()), ("clf", XGBClassifier())]
    )
    pipe_with_cleanlab.fit(X_noisy, y_noisy)
    with_cleanlab_score = pipe_with_cleanlab.score(X_clean, y_clean)

    pipe_without_cleanlab = Pipeline([("clf", XGBClassifier())])
    pipe_without_cleanlab.fit(X_noisy, y_noisy)
    without_cleanlab_score = pipe_without_cleanlab.score(X_clean, y_clean)

    assert float(with_cleanlab_score) > float(without_cleanlab_score)
