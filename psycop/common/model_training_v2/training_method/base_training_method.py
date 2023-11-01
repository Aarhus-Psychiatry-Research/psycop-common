# Implement this object for cross-validation, split-validation

from dataclasses import dataclass
from typing import Protocol

from ..loggers.base_logger import BaselineLogger
from ..presplit_preprocessing.pipeline import PreprocessingPipeline
from ..presplit_preprocessing.polars_frame import PolarsFrame
from ..problem_type.eval_dataset_base import BaseEvalDataset
from ..problem_type.problem_type_base import ProblemType


@dataclass(frozen=True)
class TrainingResult:
    metric: float
    eval_dataset: BaseEvalDataset


class TrainingMethod(Protocol):
    def train(self) -> TrainingResult:
        ...


class CrossValidatorTrainer(TrainingMethod):
    def __init__(
        self,
        data: PolarsFrame,
        preprocessing_pipeline: PreprocessingPipeline,
        problem_type: ProblemType,
        n_splits: int,
        logger: BaselineLogger,
    ):
        ...

    def train(self) -> TrainingResult:
        ...


class SplitTrainer(TrainingMethod):
    def __init__(
        self,
        training_data: PolarsFrame,
        validation_data: PolarsFrame,
        preprocessing_pipeline: PreprocessingPipeline,
        problem_type: ProblemType,
        logger: BaselineLogger,
    ):
        ...

    def train(self) -> TrainingResult:
        ...
