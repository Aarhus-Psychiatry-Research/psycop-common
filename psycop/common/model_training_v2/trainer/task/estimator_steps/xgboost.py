from typing import Any, Literal

import numpy as np
import optuna
from xgboost import XGBClassifier

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.task.model_step import ModelStep

from ....hyperparameter_suggester.suggesters.base_suggester import Suggester
from ....hyperparameter_suggester.suggesters.suggester_spaces import (
    FloatSpace,
    FloatSpaceT,
    IntegerSpace,
    IntegerspaceT,
)


@BaselineRegistry.estimator_steps.register("xgboost")
def xgboost_classifier_step(
    alpha: float = 0,
    reg_lambda: float = 1,
    max_depth: int = 3,
    learning_rate: float = 0.3,
    gamma: float = 0,
    tree_method: Literal["auto", "gpu_hist"] = "gpu_hist",
    grow_policy: Literal["depthwise", "lossguide"] = "depthwise",
    n_estimators: int = 100,
) -> ModelStep:
    """Initialize XGBClassifier model with hparams specified as kwargs.
    The 'missing' hyperparameter specifies the value to be treated as missing and is set to np.nan by default.
    """
    return (
        "classifier",
        XGBClassifier(
            alpha=alpha,
            gamma=gamma,
            learning_rate=learning_rate,
            max_depth=max_depth,
            missing=np.nan,
            n_estimators=n_estimators,
            reg_lambda=reg_lambda,
            tree_method=tree_method,
            grow_policy=grow_policy,
        ),
    )


@BaselineRegistry.estimator_steps_suggesters.register("xgboost_suggester")
class XGBoostSuggester(Suggester):
    def __init__(
        self,
        n_estimators: IntegerspaceT = (100, 1200, True),
        alpha: FloatSpaceT = (1e-8, 0.1, True),
        reg_lambda: FloatSpaceT = (1e-8, 1.0, True),
        max_depth: IntegerspaceT = (3, 8, True),
        learning_rate: FloatSpaceT = (1e-8, 1, True),
        gamma: FloatSpaceT = (1e-8, 0.001, True),
    ):
        # A little annoying, can be auto-generated using introspection of the annotations/types. E.g. added to the `Suggester` class. But this is fine for now.
        self.n_estimators = IntegerSpace.from_list_or_mapping(n_estimators)
        self.alpha = FloatSpace.from_list_or_mapping(alpha)
        self.reg_lambda = FloatSpace.from_list_or_mapping(reg_lambda)
        self.max_depth = IntegerSpace.from_list_or_mapping(max_depth)
        self.learning_rate = FloatSpace.from_list_or_mapping(learning_rate)
        self.gamma = FloatSpace.from_list_or_mapping(gamma)

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        # The same goes forthis, can be auto-generated.
        return {
            "@estimator_steps": "xgboost",
            "n_estimators": self.n_estimators.suggest(trial, name="n_estimators"),
            "alpha": self.alpha.suggest(trial, name="alpha"),
            "reg_lambda": self.reg_lambda.suggest(trial, name="reg_lambda"),
            "max_depth": self.max_depth.suggest(trial, name="max_depth"),
            "learning_rate": self.learning_rate.suggest(trial, name="learning_rate"),
            "gamma": self.gamma.suggest(trial, name="gamma"),
        }
