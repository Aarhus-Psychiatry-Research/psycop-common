from pathlib import Path
from typing import Callable

from confection import Config

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger
from psycop.common.model_training_v2.trainer.task.base_metric import CalculatedMetric


@BaselineRegistry.loggers.register("multi_logger")
class MultiLogger(BaselineLogger):
    """This logger allows combining multiple loggers. E.g. you can combine a TerminalLogger and a FileLogger."""

    def __init__(self, *args: BaselineLogger):
        self.loggers = args

    def _run_on_loggers(self, func: Callable[[BaselineLogger], None]):
        """Run a function on all loggers."""
        for logger in self.loggers:
            func(logger)

    def info(self, message: str):
        self._run_on_loggers(lambda logger: logger.info(message))

    def good(self, message: str):
        self._run_on_loggers(lambda logger: logger.good(message))

    def warn(self, message: str):
        self._run_on_loggers(lambda logger: logger.warn(message))

    def fail(self, message: str):
        self._run_on_loggers(lambda logger: logger.fail(message))

    def log_metric(self, metric: CalculatedMetric):
        self._run_on_loggers(lambda logger: logger.log_metric(metric=metric))

    def log_config(self, config: Config):
        self._run_on_loggers(lambda logger: logger.log_config(config=config))

    def log_artifact(self, local_path: Path):
        self._run_on_loggers(lambda logger: logger.log_artifact(local_path=local_path))
