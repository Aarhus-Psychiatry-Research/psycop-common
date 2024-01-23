import random
import string
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import confection
import mlflow

from psycop.common.global_utils.config_utils import (
    flatten_nested_dict,
    replace_symbols_in_dict_keys,
)
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger
from psycop.common.model_training_v2.trainer.task.base_metric import CalculatedMetric


def sanitise_dict_keys(d: dict[str, Any]) -> dict[str, Any]:
    return replace_symbols_in_dict_keys(d=d, symbol2replacement={"@": "", "*": "_"})


@BaselineRegistry.loggers.register("mlflow_logger")
class MLFlowLogger(BaselineLogger):
    def __init__(
        self, experiment_name: str, tracking_uri: str = "http://exrhel0371.it.rm.dk:5050"
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self.mlflow_experiment = mlflow.set_experiment(experiment_name=experiment_name)
        self.experiment_id = self.mlflow_experiment.experiment_id
        self._log_str = ""

    def _append_log_str(self, prefix: str, message: str):
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_str += f"\n[{timestamp_str}] {prefix}: {message}"

    def _log_text_as_artifact(self, text: str, filename: str, remote_dir: str | None = None):
        """Log text as an artifact.

        This is a workaround for MLFlow not supporting logging text directly.
        Note that multiple logs to the same remote_path will overwrite each other."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = Path(tmp_dir) / filename
            tmp_file.write_text(text)
            mlflow.log_artifact(local_path=str(tmp_file), artifact_path=remote_dir)

    def _log(self, prefix: str, message: str):
        """MLFLow supports logging metrics, parameters, datasets and artifacts. Since a log message is neither,
        the workaround is to create a log file, and then log that as an artifact."""
        self._append_log_str(prefix, message)
        self._log_text_as_artifact(text=self._log_str, filename="log.txt")

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
        flattened_config = flatten_nested_dict(config)
        mlflow.log_params(sanitise_dict_keys(flattened_config))
        self._log_text_as_artifact(confection.Config(config).to_str(), filename="config.cfg")

    def log_artifact(self, local_path: Path) -> None:
        mlflow.log_artifact(local_path=local_path.__str__())
