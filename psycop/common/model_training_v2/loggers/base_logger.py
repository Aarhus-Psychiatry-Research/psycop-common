from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import mlflow
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
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "http://exrhel0371.it.rm.dk:5050",
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name=experiment_name)

        self._log_str = ""

    def _append_log_str(self, prefix: str, message: str):
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_str += f"\n[{timestamp_str}] {prefix}: {message}"

    def _log(self, prefix: str, message: str):
        """MLFLow supports logging metrics, parameters, datasets and artifacts. Since a log message is neither,
        the workaround is to create a log file, and then log that as an artifact."""

        self._append_log_str(prefix, message)

        tmp_log_path = Path("tmp_log.txt")

        # Write log temporarily to disk
        with tmp_log_path.open("w") as f:
            f.write(self._log_str)

        # Overwrites any previously saved artifacts
        mlflow.log_artifact(local_path=tmp_log_path.__str__(), artifact_path="log.txt")

        # Delete the log on disk
        tmp_log_path.unlink()

    def info(self, message: str) -> None:
        self._log(prefix="INFO", message=message)

    def good(self, message: str) -> None:
        self._log(prefix="GOOD", message=message)

    def warn(self, message: str) -> None:
        self._log(prefix="WARN", message=message)

    def fail(self, message: str) -> None:
        self._log(prefix="FAIL", message=message)

    def log_metric(self, metric: CalculatedMetric) -> None:
        mlflow.log_metric(key=metric.name, value=metric.value)

    def log_config(self, config: dict[str, Any]):
        config = flatten_nested_dict(config)
        for k, v in config.items():
            mlflow.log_param(key=k, value=v)
