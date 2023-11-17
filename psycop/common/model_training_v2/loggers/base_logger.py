from typing import Any, Protocol, runtime_checkable

from psycop.common.model_training_v2.trainer.task.base_metric import (
    CalculatedMetric,
)


@runtime_checkable
class BaselineLogger(Protocol):
    def info(self, message: str) -> None:
        ...

    def good(self, message: str) -> None:
        ...

    def warn(self, message: str) -> None:
        ...

    def fail(self, message: str) -> None:
        ...

    def log_metric(self, metric: CalculatedMetric) -> None:
        ...

    def log_config(self, config: dict[str, Any]) -> None:
        ...


