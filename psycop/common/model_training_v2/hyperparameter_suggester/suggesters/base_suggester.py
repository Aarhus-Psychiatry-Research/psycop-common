from typing import Any, Protocol, runtime_checkable

import optuna


@runtime_checkable
class Suggester(Protocol):
    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]: ...
