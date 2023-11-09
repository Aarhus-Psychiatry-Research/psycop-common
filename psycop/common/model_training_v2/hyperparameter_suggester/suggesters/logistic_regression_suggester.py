from dataclasses import dataclass
from typing import Any, Protocol

import optuna


@dataclass(frozen=True)
class FloatSpace:
    low: float
    high: float
    logarithmic: bool

    def suggest(self, trial: optuna.Trial, name: str) -> float:
        return trial.suggest_float(
            name=name, low=self.low, high=self.high, log=self.logarithmic
        )

class Suggester(Protocol):
    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        ...

class LogisticRegressionSuggester(Suggester):
    def __init__(self, C: FloatSpace, l1_ratio: FloatSpace):
        self.C = C
        self.l1_ratio = l1_ratio

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "logistic_regression": {
                "@estimator_steps": "logistic_regression",
                "C": self.C.suggest(trial, "C"),
                "l1_ratio": self.l1_ratio.suggest(trial, "l1_ratio"),
            }
        }
