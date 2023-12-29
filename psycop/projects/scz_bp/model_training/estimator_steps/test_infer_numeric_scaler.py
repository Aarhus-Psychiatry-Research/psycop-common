import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from psycop.projects.scz_bp.model_training.estimator_steps.infer_numeric_standardisation import (
    InferNumericStandardScaler,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "numeric1": np.random.normal(5, 2, size=50),
            "numeric2": np.random.normal(10, 2, size=50),
            "binary_categorical": [0, 0, 0, 1, 1] * 10,
            "categorical": [0, 1, 2, 3, 4] * 10,
        }
    )


@pytest.fixture
def custom_scaler() -> InferNumericStandardScaler:
    return InferNumericStandardScaler()


@pytest.fixture
def standard_scaler() -> StandardScaler:
    return StandardScaler()


def test_numeric_and_non_numeric_columns(
    sample_df: pd.DataFrame,
    custom_scaler: InferNumericStandardScaler,
    standard_scaler: StandardScaler,
):
    scaled_df = custom_scaler.fit_transform(sample_df)
    pd.testing.assert_series_equal(scaled_df["categorical"], sample_df["categorical"])
    pd.testing.assert_series_equal(
        scaled_df["binary_categorical"], sample_df["binary_categorical"]
    )

    standard_scaled = standard_scaler.fit_transform(sample_df[["numeric1", "numeric2"]])
    assert np.allclose(scaled_df[["numeric1", "numeric2"]], standard_scaled)


def test_output_shape(
    sample_df: pd.DataFrame, custom_scaler: InferNumericStandardScaler
):
    scaled_df = custom_scaler.fit_transform(sample_df)
    assert scaled_df.shape == sample_df.shape


def test_fit_idempotency(
    sample_df: pd.DataFrame, custom_scaler: InferNumericStandardScaler
):
    custom_scaler.fit(sample_df)
    mean1, var1 = custom_scaler.mean_, custom_scaler.var_

    custom_scaler.fit(sample_df)
    mean2, var2 = custom_scaler.mean_, custom_scaler.var_

    assert np.allclose(mean1, mean2)  # type: ignore
    assert np.allclose(var1, var2)  # type: ignore
