from typing import Protocol


class Logger(Protocol):
    run_name: str
    metrics: list[dict[str, float]]

    def __init__(self, project_name: str, run_name: str):
        ...

    def log_metrics(self, metrics: dict[str, float]) -> None:
        ...
