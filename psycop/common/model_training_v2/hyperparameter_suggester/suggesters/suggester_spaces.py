from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import optuna

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.base_suggester import (
    Suggester,
)

FloatSpaceT = Mapping[str, float | bool]
# Used when specifying mappings in the confection .cfg, which is then immediately cast to a FloatSpace
# As such, requires keys that correspond to the FloatSpace's attributes (see below)


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

    @classmethod
    def from_mapping(cls: type["FloatSpace"], mapping: FloatSpaceT) -> "FloatSpace":
        return cls(
            low=mapping["low"],
            high=mapping["high"],
            logarithmic=mapping["logarithmic"],  # type: ignore
        )


@dataclass(frozen=True)
class CategoricalSpace:
    choices: Sequence[optuna.distributions.CategoricalChoiceType]

    def suggest(self, trial: optuna.Trial, name: str) -> Any:
        return trial.suggest_categorical(name=name, choices=self.choices)


@BaselineRegistry.suggesters.register("mock_suggester")
class MockSuggester(Suggester):
    """Suggester used only for tests. Ensures tests only break if the interface breaks, not because of implementation details in e.g. LogisticRegression."""

    def __init__(
        self,
        value_low: float,
        value_high: float,
        log: bool,
        suggested_key: str = "mock_value",
        optuna_key: str = "mock_suggester",
    ):
        self.value = FloatSpace(low=value_low, high=value_high, logarithmic=log)
        self.optuna_key = optuna_key
        self.suggested_key = suggested_key

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        return {self.suggested_key: self.value.suggest(trial, self.optuna_key)}