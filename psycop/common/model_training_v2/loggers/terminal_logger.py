from pathlib import Path

import wasabi
from confection import Config

from psycop.common.global_utils.config_utils import flatten_nested_dict
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
        config = flatten_nested_dict(config)  # type: ignore # Config is a subclass of dict so false positive
        cfg_str = "\n".join([f"{k}: {v}" for k, v in config.items()])
        self._l.info(cfg_str)

    def log_artifact(self, local_path: Path) -> None:
        self.good(
            f"""Logging artifact at {local_path}.
    NOTE: TerminalLogger does not log the artifact anywhere, but if you have other loggers defined, their methods have been called as well.""",
        )
