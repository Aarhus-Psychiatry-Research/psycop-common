from typing import Optional

import pandas as pd
from deepchecks.utils.type_inference import infer_categorical_features
from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.task.model_step import ModelStep


class InferNumericStandardScaler(StandardScaler):
    """Standardize features by removing the mean and scaling to unit variance.
    This estimator automatically detects numeric columns and only standardizes them.
    Uses deepcheck's type inference to infer numeric columns, and sklearn's StandardScaler
    """

    def __init__(
        self,
        *,
        copy: bool = True,
        with_mean: bool = True,
        with_std: bool = True,
    ):
        super().__init__(
            copy=copy,
            with_mean=with_mean,
            with_std=with_std,
        )
        self.numeric_cols: list[str] = []

    def _infer_numeric_cols(self, X: pd.DataFrame) -> list[str]:
        """Infer numeric columns from the input dataframe using deepcheck's
        type inference. Could potentialy be simplified by interpreting colunms
        that are integers as categorical."""
        categorical_cols = infer_categorical_features(X)
        numeric_cols = list(set(X.columns) - set(categorical_cols))  # type: ignore
        return numeric_cols

    def fit(  # type: ignore
        self,
        X: pd.DataFrame,
        y: None = None,
        sample_weight: ArrayLike | None = None,
    ) -> "InferNumericStandardScaler":
        """Fit the StandardScaler to the inferred numeric columns.
        y is ignored. See https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler.fit
        for details"""
        self.numeric_cols = self._infer_numeric_cols(X)

        # Fit only on numeric columns
        super().fit(X=X[self.numeric_cols], y=y, sample_weight=sample_weight)
        return self

    def transform(self, X: pd.DataFrame, copy: Optional[bool] = None) -> pd.DataFrame:  # type: ignore
        X_transformed = X.copy()

        X_transformed[self.numeric_cols] = super().transform(
            X[self.numeric_cols],
            copy=copy,
        )
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: None = None) -> pd.DataFrame:  # type: ignore
        return self.fit(X, y).transform(X)


@BaselineRegistry.estimator_steps.register("numeric_z_score_standardisation")
def z_score_standardisation() -> ModelStep:
    return (
        "numeric_z_score_standardisation",
        InferNumericStandardScaler(),
    )
