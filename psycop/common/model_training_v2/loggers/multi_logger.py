from typing import Any

from functionalpy import Seq

from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger
from psycop.common.model_training_v2.metrics.base_metric import CalculatedMetric


class MultiLogger(BaselineLogger):
    """This logger allows combining multiple loggers. E.g. you can combine a TerminalLogger and a FileLogger."""
    def __init__(self, *args: BaselineLogger):
        self.loggers = args
    
    def info(self, message: str):
        Seq(self.loggers).map(lambda logger: logger.info(message)).to_list()

    def good(self, message: str):
        Seq(self.loggers).map(lambda logger: logger.good(message)).to_list()

    def warn(self, message: str):
        Seq(self.loggers).map(lambda logger: logger.warn(message)).to_list()

    def fail(self, message: str):
        Seq(self.loggers).map(lambda logger: logger.fail(message)).to_list()

    def log_metric(self, metric: CalculatedMetric):
        Seq(self.loggers).map(lambda logger: logger.log_metric(metric=metric)).to_list()

    def log_config(self, config: dict[str, Any]):
        Seq(self.loggers).map(lambda logger: logger.log_config(config=config)).to_list()
    