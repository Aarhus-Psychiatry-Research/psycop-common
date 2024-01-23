import sys

import pandas as pd
import pytest
from imblearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

if sys.modules.get("synthcity") is None:
    pytest.skip("requires the synthcity library", allow_module_level=True)

from psycop.projects.scz_bp.model_training.estimator_steps.synth_data_augmentation import (
    SyntheticDataAugmentation,
)


@pytest.fixture()
def sample_data() -> tuple[pd.DataFrame, pd.Series]:  # type: ignore
    X, y = make_classification(
        n_classes=2, weights=(0.9, 0.1), flip_y=0, n_features=5, n_samples=50, random_state=42
    )
    return pd.DataFrame(X), pd.Series(y, name="target")


def test_initialization():
    augmenter = SyntheticDataAugmentation("ddpm", {"n_iter": 10}, prop_augmented=0.5)
    assert augmenter.model_name == "ddpm"
    assert "n_iter" in augmenter.model_params
    assert augmenter.prop_augmented == 0.5


def test_prop_augmented(sample_data: tuple[pd.DataFrame, pd.Series]):  # type: ignore
    X, y = sample_data
    aug = SyntheticDataAugmentation("ddpm", prop_augmented=0.5, model_params={"n_iter": 1})
    X_res, y_res = aug.fit_resample(X, y)  # type: ignore
    assert len(X_res) == int(1.5 * len(X))  # check if 50% more samples are added
    assert len(y_res) == int(1.5 * len(y))


def test_minority_strategy(sample_data: tuple[pd.DataFrame, pd.Series]):  # type: ignore
    X, y = sample_data
    # should only add cases of the minority (assumed to be 1) class
    target_aug = SyntheticDataAugmentation(
        "ddpm", sampling_strategy="minority", prop_augmented=0.5, model_params={"n_iter": 1}
    )
    X_res_minority, y_res_minority = target_aug.fit_resample(X, y)  # type: ignore
    n_minority_in_X: int = len(y[y == 1])
    n_minority_in_X_target_aug = len(y_res_minority[y_res_minority == 1])  # type: ignore

    assert n_minority_in_X_target_aug == n_minority_in_X + 0.5 * X.shape[0]


def test_synth_data_augmentation_in_pipeline(
    sample_data: tuple[pd.DataFrame, pd.Series],  # type: ignore
):
    X, y = sample_data
    aug = SyntheticDataAugmentation("ddpm", prop_augmented=0.5, model_params={"n_iter": 1})
    pipe_with_aug = Pipeline([("aug", aug), ("clf", XGBClassifier())])
    pipe_with_aug.fit(X, y)
    with_aug_score = pipe_with_aug.score(X, y)

    pipe = Pipeline([("clf", XGBClassifier())])
    pipe.fit(X, y)
    score = pipe.score(X, y)

    # score should at least be different
    assert with_aug_score != score
