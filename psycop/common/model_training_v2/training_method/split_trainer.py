from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger
from psycop.common.model_training_v2.presplit_preprocessing.pipeline import (
    PreprocessingPipeline,
)
from psycop.common.model_training_v2.presplit_preprocessing.polars_frame import (
    PolarsFrame,
)
from psycop.common.model_training_v2.problem_type.problem_type_base import ProblemType
from psycop.common.model_training_v2.training_method.base_training_method import (
    TrainingMethod,
    TrainingResult,
)


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
