from typing import Any, Literal

import numpy as np
import pandas as pd
from cleanlab.classification import CleanLearning
from imblearn.base import BaseSampler
from xgboost import XGBClassifier

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.task.model_step import ModelStep


class CleanlabProcessing(BaseSampler):
    """Apply data-centric processing using cleanlab to remove "bad" data points."""

    def __init__(
        self,
    ):
        self.sampling_strategy = "all"  # imblearn internal
        self._sampling_type = "ensemble"  # imblearn internal
        self.n_label_issues = 0

    def _fit_resample(  # type: ignore
        self,
        X: np.ndarray,  # type: ignore
        y: np.ndarray,  # type: ignore
    ) -> tuple[np.ndarray, np.ndarray]:  # type: ignore
        """Apply cleanlab to the training data and data points with label issues"""
        X_copy = X.copy()
        y_copy = y.copy()

        model = XGBClassifier()
        cl = CleanLearning(model)
        cl.fit(X_copy, y_copy)
        label_issues = cl.get_label_issues()
        if label_issues is None:
            return X_copy, y_copy

        label_error_indices = np.where(label_issues["is_label_issue"])[0]
        self.n_label_issues = len(label_error_indices)
        # drop label issues indices from X and y
        X_copy = np.delete(X_copy, label_error_indices, axis=0)
        y_copy = np.delete(y_copy, label_error_indices, axis=0)

        return X_copy, y_copy


@BaselineRegistry.estimator_steps.register("cleanlab_processing")
def cleanlab_processing_step() -> ModelStep:
    return ("synthetic_data_augmentation", CleanlabProcessing())
