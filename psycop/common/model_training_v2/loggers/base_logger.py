from typing import Any, Protocol


class BaselineLogger(Protocol):
    def log_metric(self, name: str, value: float) -> None:
        ...

    def log_config(self, config: dict[str, Any]) -> None:
        ...
