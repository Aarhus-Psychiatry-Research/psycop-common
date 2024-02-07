import logging
import shutil
from pathlib import Path

from confection import Config

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger
from psycop.common.model_training_v2.trainer.task.base_metric import CalculatedMetric


@BaselineRegistry.loggers.register("disk_logger")
class DiskLogger(BaselineLogger):
    def __init__(self, experiment_path: str) -> None:
        self.experiment_path = Path(experiment_path)
        self.log_path = self.experiment_path / "logs.log"
        self.cfg_log_path = self.experiment_path / "config.cfg"

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

    def log_config(self, config: Config) -> None:
        self.info(f"Logging config to cfg file at {self.cfg_log_path}")
        config.to_disk(self.cfg_log_path)

    def _setup_logging_module(self) -> logging.Logger:
        logger = logging.getLogger("DiskLogger")
        logger.setLevel(logging.DEBUG)

        self.log_path.parent.mkdir(exist_ok=True, parents=True)

        # Create file handler
        fh = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(fh)

        logger.info("Disk logger initialized")
        return logger

    def log_artifact(self, local_path: Path) -> None:
        shutil.copy(local_path, (self.experiment_path / local_path.name))
