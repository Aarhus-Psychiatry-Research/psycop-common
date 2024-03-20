from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import optuna

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.base_suggester import (
    Suggester,
)

FloatspaceMappingT = Mapping[str, float | bool]
FloatSpaceSequenceT = Sequence[float | bool]
FloatSpaceT = FloatspaceMappingT | FloatSpaceSequenceT
# Used when specifying mappings in the confection .cfg, which is then immediately cast to a FloatSpace
# For the mapping, requires keys that correspond to the FloatSpace's attributes (see below)
# For the sequence, requres a float, float, int


@dataclass(frozen=True)
class FloatSpace:
    low: float
    high: float
    logarithmic: bool

    def __post_init__(self):
        if self.low >= self.high:
            raise ValueError(
                f"Invalid FloatSpace: low value {self.low} must be less than high value {self.high}"
            )
        if self.logarithmic and (self.low <= 0 or self.high <= 0):
            raise ValueError(
                "Invalid FloatSpace: The logarithm is undefined for values less than or equal to 0, so all values must be greater than 0."
            )

    def suggest(self, trial: optuna.Trial, name: str) -> float:
        return trial.suggest_float(name=name, low=self.low, high=self.high, log=self.logarithmic)

    @classmethod
    def from_mapping(cls: type["FloatSpace"], mapping: FloatspaceMappingT) -> "FloatSpace":
        return cls(
            low=mapping["low"],
            high=mapping["high"],
            logarithmic=mapping["logarithmic"],  # type: ignore
        )

    @classmethod
    def from_list(cls: type["FloatSpace"], sequence: FloatSpaceSequenceT) -> "FloatSpace":
        return cls(
            low=sequence[0],
            high=sequence[1],
            logarithmic=sequence[2],  # type: ignore
        )

    @classmethod
    def from_list_or_mapping(
        cls: type["FloatSpace"], sequence_or_mapping: FloatSpaceT
    ) -> "FloatSpace":
        if isinstance(sequence_or_mapping, Mapping):
            return cls.from_mapping(sequence_or_mapping)
        return cls.from_list(sequence_or_mapping)


IntegerspaceMappingT = Mapping[str, int | bool]
IntegerspaceSequenceT = Sequence[int | bool]
IntegerspaceT = IntegerspaceMappingT | IntegerspaceSequenceT


@dataclass(frozen=True)
class IntegerSpace:
    low: int
    high: int
    logarithmic: bool

    def __post_init__(self):
        if self.low >= self.high:
            raise ValueError(
                f"Invalid IntegerSpace: low value {self.low} must be less than high value {self.high}"
            )
        if self.logarithmic and (self.low <= 0 or self.high <= 0):
            raise ValueError(
                "Invalid IntegerSpace: The logarithm is undefined for values less than or equal to 0, so all values must be greater than 0."
            )

    def suggest(self, trial: optuna.Trial, name: str) -> float:
        return trial.suggest_int(name=name, low=self.low, high=self.high, log=self.logarithmic)

    @classmethod
    def from_mapping(cls: type["IntegerSpace"], mapping: IntegerspaceMappingT) -> "IntegerSpace":
        return cls(
            low=mapping["low"],
            high=mapping["high"],
            logarithmic=mapping["logarithmic"],  # type: ignore
        )

    @classmethod
    def from_list(cls: type["IntegerSpace"], sequence: IntegerspaceSequenceT) -> "IntegerSpace":
        return cls(
            low=sequence[0],
            high=sequence[1],
            logarithmic=sequence[2],  # type: ignore
        )

    @classmethod
    def from_list_or_mapping(
        cls: type["IntegerSpace"], sequence_or_mapping: IntegerspaceT
    ) -> "IntegerSpace":
        if isinstance(sequence_or_mapping, Mapping):
            return cls.from_mapping(sequence_or_mapping)
        return cls.from_list(sequence_or_mapping)


CategoricalSpaceT = Sequence[optuna.distributions.CategoricalChoiceType]


@dataclass(frozen=True)
class CategoricalSpace:
    choices: CategoricalSpaceT

    def suggest(self, trial: optuna.Trial, name: str) -> Any:
        return trial.suggest_categorical(name=name, choices=self.choices)



@dataclass
class SingleValue:
    """If you don't want to search across all possible hparams"""
    val: str | float

    def suggest(self, trial: optuna.Trial, name: str) -> Any:
        return self.val

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
