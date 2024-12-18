from pathlib import Path

import wasabi
from confection import Config
from polars import DataFrame
#from rich.pretty import pprint

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger
from psycop.common.model_training_v2.trainer.task.base_metric import CalculatedMetric


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
        self._l.info(f"{metric.name}: {round(metric.value, 2)}")

    def log_config(self, config: Config) -> None:
        self._l.divider("Logging config")
        print(config)

    def log_artifact(self, local_path: Path) -> None:
        self.good(
            f"""Logging artifact from {local_path}.
    NOTE: TerminalLogger does not log the artifact anywhere, but if you have other loggers defined, their methods have been called as well."""
        )

    def log_dataset(self, dataframe: DataFrame, filename: str) -> None:  # noqa: ARG002
        self.good(
            "NOTE: TerminalLogger does not log the dataset anywhere, but if you have other loggers defined, their methods have been called as well."
        )
