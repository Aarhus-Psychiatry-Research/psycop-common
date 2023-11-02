from typing import Any, Protocol


class BaselineLogger(Protocol):
    def good(self, message: str) -> None:
        ...

    def warn(self, message: str) -> None:
        ...

    def fail(self, message: str) -> None:
        ...

    def log_metric(self, name: str, value: float) -> None:
        ...

    def log_config(self, config: dict[str, Any]) -> None:
        ...
