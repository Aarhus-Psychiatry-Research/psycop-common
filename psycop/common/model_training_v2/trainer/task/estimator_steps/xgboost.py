from typing import Literal

import numpy as np
from xgboost import XGBClassifier

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.task.model_step import ModelStep


@BaselineRegistry.estimator_steps.register("xgboost")
def xgboost_classifier_step(
    tree_method: Literal["auto", "gpu_hist"] = "gpu_hist",
    n_estimators: int = 100,
    max_depth: int = 3,
) -> ModelStep:
    """Initialize XGBClassifier model with hparams specified as kwargs.
    The 'missing' hyperparameter specifies the value to be treated as missing and is set to np.nan by default.
    """
    return (
        "xgboost",
        XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, tree_method=tree_method, missing=np.nan
        ),
    )
