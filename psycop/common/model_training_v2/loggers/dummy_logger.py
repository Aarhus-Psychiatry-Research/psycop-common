from pathlib import Path

from confection import Config

from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger
from psycop.common.model_training_v2.trainer.task.base_metric import CalculatedMetric


class DummyLogger(BaselineLogger):
    def info(self, message: str) -> None:
        pass

    def good(self, message: str) -> None:
        pass

    def warn(self, message: str) -> None:
        pass

    def fail(self, message: str) -> None:
        pass

    def log_metric(self, metric: CalculatedMetric) -> None:
        pass

    def log_config(self, config: Config) -> None:
        pass

    def log_artifact(self, local_path: Path) -> None:
        pass
