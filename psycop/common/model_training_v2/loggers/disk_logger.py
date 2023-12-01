import json
import logging
from pathlib import Path
from typing import Any

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger
from psycop.common.model_training_v2.trainer.task.base_metric import CalculatedMetric


@BaselineRegistry.loggers.register("disk_logger")
class TerminalLogger(BaselineLogger):
    def __init__(self, experiment_path: str) -> None:
        self.experiment_path = Path(experiment_path)
        self.logger_path = self.experiment_path / "logs.log"
        self.cfg_log_path = self.experiment_path / "config.json"

        self._l = self._setup_logging_module()

    def info(self, message: str) -> None:
        self._l.info(message)

    def good(self, message: str) -> None:
        self._l.info(message)

    def warn(self, message: str) -> None:
        self._l.warn(message)

    def fail(self, message: str) -> None:
        self._l.error(message)

    def log_metric(self, metric: CalculatedMetric) -> None:
        self._l.info(f"{metric.name}: {round(metric.value, 2)}")

    def log_config(self, config: dict[str, Any]) -> None:
        self.info(f"Logging config to json file at {self.cfg_log_path}")
        with self.cfg_log_path.open("w") as f:
            json.dump(config, f)

    def _setup_logging_module(self) -> logging.Logger:
        logging.basicConfig(
            filename=self.logger_path,
            level=logging.DEBUG,
            encoding="utf-8",
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger_path.mkdir(exist_ok=True, parents=True)

        logging.info("Disk logger initialized")
        return logging.getLogger()
