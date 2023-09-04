from typing import Protocol


class Logger(Protocol):
    def __init__(self, project_name: str):
        ...

    def log_metrics(self, metrics: dict[str, float]) -> None:
        ...

    @property
    def run_name(self) -> str:
        ...
