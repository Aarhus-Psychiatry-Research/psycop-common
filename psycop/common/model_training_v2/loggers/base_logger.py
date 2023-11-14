from typing import Any, Protocol, runtime_checkable

import wasabi

from psycop.common.global_utils.config_utils import flatten_nested_dict
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
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


@BaselineRegistry.loggers.register("terminal_logger")
class TerminalLogger(BaselineLogger):
    def __init__(self) -> None:
        self._l = wasabi.Printer(timestamp=True)

    def info(self, message: str) -> None:
        self._l.info(message)

    def good(self, message: str) -> None:
        self._l.good(message)

    def warn(self, message: str) -> None:
        self._l.warn(message)

    def fail(self, message: str) -> None:
        self._l.fail(message)

    def log_metric(self, metric: CalculatedMetric) -> None:
        self._l.divider(f"Logging metric {metric.name}")
        self._l.info(f"{metric.name}: {metric.value}")

    def log_config(self, config: dict[str, Any]) -> None:
        self._l.divider("Logging config")
        config = flatten_nested_dict(config)
        cfg_str = "\n".join([f"{k}: {v}" for k, v in config.items()])
        self._l.info(cfg_str)

class MLFlowLogger(BaselineLogger):
    def __init__(self, tracking_uri: str) -> None:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        self._mlflow = mlflow