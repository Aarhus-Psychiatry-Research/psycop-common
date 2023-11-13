from typing import Any, Protocol

import optuna


class Suggester(Protocol):
    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        ...
