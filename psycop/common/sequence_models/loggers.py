from typing import Protocol


class Logger(Protocol):
    def log_hyperparams(self, hyperparams: dict[str, float | str]) -> None:
        ...

    def log_metrics(self, metrics: dict[str, float]) -> None:
        ...
