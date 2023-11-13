from dataclasses import dataclass
from typing import Any

import optuna

from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.base_suggester import (
    Suggester,
)

FloatSpaceT = tuple[float, float, bool]


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


class LogisticRegressionSuggester(Suggester):
    # XXX: Should we move this to the logistic regression creator, and then have the creator also subclass suggester?
    # Perhaps that makes it hard to decorate, though?

    # XXX: When using confection, this class becomes _much_ easier to initialise if initialised with tuples of (low, high, logarithmic). Then we can recast to FloatSpace. Otherwise, we will have to initialise a dataclass every time. Do we agree here?
    def __init__(self, C: FloatSpaceT, l1_ratio: FloatSpaceT):
        self.C = FloatSpace(low=C[0], high=C[1], logarithmic=C[2])
        self.l1_ratio = FloatSpace(low=l1_ratio[0], high=l1_ratio[1], logarithmic=l1_ratio[2])

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "logistic_regression": {
                "@estimator_steps": "logistic_regression",
                "C": self.C.suggest(trial, "C"),
                "l1_ratio": self.l1_ratio.suggest(trial, "l1_ratio"),
            },
        }
