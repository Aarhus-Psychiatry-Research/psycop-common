from dataclasses import dataclass
from typing import Any

import optuna

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.base_suggester import (
    Suggester,
)

FloatSpaceT = list[float | bool]


@dataclass(frozen=True)
class FloatSpace:
    low: float
    high: float
    logarithmic: bool

    def suggest(self, trial: optuna.Trial, name: str) -> float:
        return trial.suggest_float(
            name=name,
            low=self.low,
            high=self.high,
            log=self.logarithmic,
        )


@BaselineRegistry.estimator_steps.register("mock_suggester")
class MockSuggester(Suggester):
    """Suggester used only for tests. Ensures tests only break if the interface breaks, not because of implementation details in e.g. LogisticRegression."""

    def __init__(self, value_low: float, value_high: float, log: bool):
        self.value = FloatSpace(low=value_low, high=value_high, logarithmic=log)

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        return {"mock_value": self.value.suggest(trial, "mock_suggester")}


@BaselineRegistry.estimator_steps.register("logistic_regression_suggester")
class LogisticRegressionSuggester(Suggester):
    # TODO: #424 Refactor so suggesters are co-located with their corresponding model steps.
    # Perhaps that makes it hard to decorate, though?

    # TODO: Can suggesters take a mapping for each argument?
    # E.g. C can be a mapping which must contain low, high and log? How do we type-hint that? A nameddict?

    def __init__(
        self,
        C_low: float,
        C_high: float,
        C_log: bool,
        l1_ratio_low: float,
        l1_ratio_high: float,
        l1_ratio_log: bool,
    ):
        self.C = FloatSpace(low=C_low, high=C_high, logarithmic=C_log)
        self.l1_ratio = FloatSpace(
            low=l1_ratio_low,
            high=l1_ratio_high,
            logarithmic=l1_ratio_log,
        )

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "logistic_regression": {
                "@estimator_steps": "logistic_regression",
                "C": self.C.suggest(trial, "C"),
                "l1_ratio": self.l1_ratio.suggest(trial, "l1_ratio"),
            },
        }
