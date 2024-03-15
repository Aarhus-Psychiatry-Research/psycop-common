import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import confection
import mlflow
from polars import DataFrame

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
@dataclass
class MLFlowLogger(BaselineLogger):
    experiment_name: str
    tracking_uri: str = "http://exrhel0371.it.rm.dk:5050"
    postpone_run_creation_to_first_log: bool = False

    def __post_init__(self) -> None:
        self._run_initialised = False

        if not self.postpone_run_creation_to_first_log:
            self._init_run()

    def _init_run(self):
        if not self._run_initialised:
            self._log_str = ""
            mlflow.set_tracking_uri(self.tracking_uri)
            # Start a new run. End a run if it already exists within the process.
            if mlflow.active_run() is not None:
                mlflow.end_run()
            self.mlflow_experiment = mlflow.set_experiment(experiment_name=self.experiment_name)
            self._run_initialised = True

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
        self._init_run()
        self._log(prefix="INFO", message=message)

    def good(self, message: str) -> None:
        self._init_run()
        self._log(prefix="GOOD", message=message)

    def warn(self, message: str) -> None:
        self._init_run()
        self._log(prefix="WARN", message=message)

    def fail(self, message: str) -> None:
        self._init_run()
        self._log(prefix="FAIL", message=message)

    def log_metric(self, metric: CalculatedMetric) -> None:
        self._init_run()
        mlflow.log_metric(key=metric.name, value=metric.value)

    def log_config(self, config: dict[str, Any]):
        self._init_run()
        flattened_config = flatten_nested_dict(config)
        mlflow.log_params(sanitise_dict_keys(flattened_config))
        self._log_text_as_artifact(confection.Config(config).to_str(), filename="config.cfg")

    def log_artifact(self, local_path: Path) -> None:
        self._init_run()
        mlflow.log_artifact(local_path=local_path.__str__())

    def log_dataset(self, dataframe: DataFrame, filename: str) -> None:
        self._init_run()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = Path(tmp_dir) / filename
            dataframe.write_parquet(tmp_file)
            mlflow.log_artifact(local_path=tmp_file.__str__())
