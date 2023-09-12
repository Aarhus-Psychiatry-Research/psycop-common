from typing import Protocol


class Logger(Protocol):
    run_name: str

    def __init__(self, project_name: str | None, run_name: str | None):
        ...

    def log_metrics(self, metrics: dict[str, float]) -> None:
        ...

    def log_hyperparams(self, params: dict[str, float | str]) -> None:
        ...
