# Implement this object for cross-validation, split-validation

from collections.abc import Sequence
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
    main_metric: CalculatedMetric
    supplementary_metrics: Sequence[CalculatedMetric] | None
    eval_dataset: BaseEvalDataset


class TrainingMethod(Protocol):
    def train(self) -> TrainingResult:
        ...


class CrossValidatorTrainer(TrainingMethod):
    def __init__(
        self,
        data: PolarsFrame,
        outcome_col_name: str,
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
        training_outcome_col_name: str,
        validation_data: PolarsFrame,
        validation_outcome_col_name: str,
        preprocessing_pipeline: PreprocessingPipeline,
        problem_type: ProblemType,
        logger: BaselineLogger,
    ):
        self.training_data = training_data
        self.training_outcome_col_name = training_outcome_col_name
        self.validation_data = validation_data
        self.validation_outcome_col_name = validation_outcome_col_name
        self.preprocessing_pipeline = preprocessing_pipeline
        self.problem_type = problem_type
        self.logger = logger

    def train(self) -> TrainingResult:
        training_data_preprocessed = self.preprocessing_pipeline.apply(
            data=self.training_data,
        )
        self.preprocessing_pipeline.apply(
            data=self.validation_data,
        )
        self.problem_type.train(
            x=training_data_preprocessed.drop(self.training_outcome_col_name),
            y=training_data_preprocessed.select(self.training_outcome_col_name),
        )
        result = self.problem_type.evaluate(
            x=self.validation_data.drop(self.validation_outcome_col_name),
            y=self.validation_data.select(self.validation_outcome_col_name),
        )

        self.logger.log_metric(name="metric", value=result.main_metric.value)

        return result
