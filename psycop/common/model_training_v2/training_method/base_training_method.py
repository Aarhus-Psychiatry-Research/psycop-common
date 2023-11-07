# Implement this object for cross-validation, split-validation

from dataclasses import dataclass
from typing import Protocol

from psycop.common.model_training_v2.metrics.base_metric import CalculatedMetric

from ..loggers.base_logger import BaselineLogger
from ..presplit_preprocessing.pipeline import PreprocessingPipeline
from ..presplit_preprocessing.polars_frame import PolarsFrame
from ..problem_type.eval_dataset_base import BaseEvalDataset
from ..problem_type.problem_type_base import ProblemType


@dataclass(frozen=True)
class TrainingResult:
    metric: CalculatedMetric
    eval_dataset: BaseEvalDataset


class TrainingMethod(Protocol):
    def train(self) -> TrainingResult:
        ...