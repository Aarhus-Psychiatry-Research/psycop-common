from typing import Any

import numpy as np
from xgboost import XGBClassifier

from psycop.common.model_training_v2.trainer.task.model_step import (
    ModelStep,
)


# TODO: Make function signature as good as for logistic regression
def xgboost_classifier_step(**kwargs: Any) -> ModelStep:
    """Initialize XGBClassifier model with hparams specified as kwargs.
    The 'missing' hyperparameter specifies the value to be treated as missing
    and is set to np.nan by default."""
    static_hyperparameters: dict[str, float] = {"missing": np.nan}
    return ("xgboost", XGBClassifier(**kwargs, **static_hyperparameters))
