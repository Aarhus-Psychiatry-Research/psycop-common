from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import optuna

from .suggesters.base_suggester import Suggester


@dataclass(frozen=True)
class SearchSpace:
    suggesters: Sequence[Suggester]

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        suggester_dict = {
            suggester.__class__.__name__: suggester for suggester in self.suggesters
        }

        # XXX: Replace __repr__ with something better
        suggester_names = list(suggester_dict.keys())
        suggester_name: str = trial.suggest_categorical("suggester", suggester_names)  # type: ignore # We know this is a string, because it must suggest from the suggester_names. Optuna should type-hint with a generic, but haven't. MB has created an issue here: https://github.com/optuna/optuna/issues/5104

        suggester = suggester_dict[suggester_name]
        return suggester.suggest_hyperparameters(trial=trial)


def hyperparameter_suggester(
    base_cfg: dict[str, Any],
    trial: optuna.Trial,
) -> dict[str, Any]:
    """Suggest hyperparameters in a config.


    Base-cfg is a config dict representing a tree. For all nodes in the tree that are of type SearchSpace, replace the node with a suggested set of a hyperparameters.

    Args:
        base_cfg: The base config, a tree of options.
        trial: The optuna trial to use for suggesting hyperparameters.
    """
    cfg = base_cfg.copy()
    process_all_values_in_dict(d=cfg, trial=trial)
    return cfg


def process_all_values_in_dict(d: dict[str, Any], trial: optuna.Trial):
    for key, value in d.items():
        match value:
            case dict():
                process_all_values_in_dict(value, trial=trial)
            case SearchSpace():
                d[key] = value.suggest_hyperparameters(trial=trial)
            case _:
                raise ValueError(
                    f"Unexpected value.\n\tType is: {type(value)}\n\tValue: {value}",
                )
