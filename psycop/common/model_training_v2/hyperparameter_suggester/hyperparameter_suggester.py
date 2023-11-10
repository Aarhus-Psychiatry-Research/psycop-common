
from dataclasses import dataclass
from typing import Any, Sequence

import optuna

from .suggesters.base_suggester import Suggester


@dataclass(frozen=True)
class SearchSpace:
    suggesters: dict[str, Suggester] 

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        suggester_names = list(self.suggesters.keys())
        suggester_name: str = trial.suggest_categorical("suggester", suggester_names) # type: ignore # We know this is a string, because it must suggest from the suggester_names. Optuna should type-hint with a generic, but haven't. MB has created an issue here: https://github.com/optuna/optuna/issues/5104

        suggester = self.suggesters[suggester_name]
        return suggester.suggest_hyperparameters(trial=trial)

        