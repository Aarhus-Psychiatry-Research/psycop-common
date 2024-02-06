from typing import Any, Literal

import optuna
from lightgbm import LGBMClassifier

from psycop.common.model_training_v2.trainer.task.model_step import ModelStep

from ....config.baseline_registry import BaselineRegistry
from ....hyperparameter_suggester.suggesters.base_suggester import Suggester
from ....hyperparameter_suggester.suggesters.suggester_spaces import (
    FloatSpace,
    FloatSpaceT,
    IntegerSpace,
    IntegerspaceT,
)


@BaselineRegistry.estimator_steps.register("lightgbm")
def lightgbm_classifier_step(
    num_leaves: int = 31,
    max_bin: int = 63,
    device_type: Literal["cpu", "gpu"] = "cpu",
    n_estimators: int = 100,
    learning_rate: float = 0.1,
) -> ModelStep:
    return (
        "lightgbm",
        LGBMClassifier(
            num_leaves=num_leaves,
            device_type=device_type,
            max_bin=max_bin,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        ),  # type: ignore
    )


@BaselineRegistry.estimator_steps_suggesters.register("lightgbm_suggester")
class LightGBMSuggester(Suggester):
    def __init__(
        self,
        num_leaves: IntegerspaceT = (5, 500, True),
        n_estimators: IntegerspaceT = (5, 500, True),
        learning_rate: FloatSpaceT = (1e-5, 0.2, True),
    ):
        # A little annoying, can be auto-generated using introspection of the annotations/types. E.g. added to the `Suggester` class. But this is fine for now.
        self.num_leaves = IntegerSpace.from_list_or_mapping(num_leaves)
        self.n_estimators = IntegerSpace.from_list_or_mapping(n_estimators)
        self.learning_rate = FloatSpace.from_list_or_mapping(learning_rate)

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        # The same goes forthis, can be auto-generated.
        return {
            "@estimator_steps": "xgboost",
            "@estimator_steps": "lightgbm",
            "num_leaves": self.num_leaves.suggest(trial, name="num_leaves"),
            "n_estimators": self.n_estimators.suggest(trial, name="n_estimators"),
            "learning_rate": self.learning_rate.suggest(trial, name="learning_rate"),
            "device_type": "gpu",
        }
