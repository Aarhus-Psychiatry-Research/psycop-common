from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow

from psycop.common.global_utils.config_utils import flatten_nested_dict
from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger
from psycop.common.model_training_v2.trainer.task.base_metric import CalculatedMetric


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
