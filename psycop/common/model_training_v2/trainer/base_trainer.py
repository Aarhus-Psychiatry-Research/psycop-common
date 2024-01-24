# Implement this object for cross-validation, split-validation
import pickle
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from psycop.common.model_training_v2.trainer.task.base_metric import CalculatedMetric

from ..loggers.base_logger import BaselineLogger
from ..loggers.supports_logger import SupportsLoggerMixin
from .task.base_task import BaselineTask


@dataclass(frozen=True)
class TrainingResult:
    metric: CalculatedMetric
    df: pl.DataFrame


class BaselineTrainer(ABC, SupportsLoggerMixin):
    task: BaselineTask

    @abstractmethod
    def train(self) -> TrainingResult:
        ...

    def _log_sklearn_pipe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = Path(tmp_dir) / "sklearn_pipe.pkl"
            with filename.open("wb") as f:
                pickle.dump(self.task.task_pipe.sklearn_pipe, f)
            self.logger.log_artifact(Path(f.name))

    def _log_main_metric(self, main_metric: CalculatedMetric) -> None:
        self.logger.log_metric(main_metric)
