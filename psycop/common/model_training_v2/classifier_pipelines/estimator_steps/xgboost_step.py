"""Specifications of models to be evaluated."""
from typing import Any

import numpy as np
from xgboost import XGBClassifier

from psycop.common.model_training_v2.classifier_pipelines.model_step import ModelStep


def xgboost_classifier_step(**kwargs: Any) -> ModelStep:
    """Initialize XGBClassifier model with hparams specified as kwargs.
    The 'missing' hyperparameter specifies the value to be treated as missing
    and is set to np.nan by default."""
    static_hyperparameters = {"missing": np.nan}
    return ("xgboost", XGBClassifier(**kwargs, **static_hyperparameters))
