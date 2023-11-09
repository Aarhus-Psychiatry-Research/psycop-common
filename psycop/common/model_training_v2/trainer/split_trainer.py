from typing import Callable

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader
from psycop.common.model_training_v2.trainer.base_trainer import (
    BaselineTrainer,
    TrainingResult,
)
from psycop.common.model_training_v2.trainer.preprocessing.pipeline import (
    PreprocessingPipeline,
)
from psycop.common.model_training_v2.trainer.preprocessing.polars_frame import (
    PolarsFrame,
)
from psycop.common.model_training_v2.trainer.task.base_task import (
    BaselineTask,
)


@BaselineRegistry.baseline_trainers.register("split_trainer")
class SplitTrainer(BaselineTrainer):
    def __init__(
        self,
        training_data: BaselineDataLoader,
        training_outcome_col_name: str,
        validation_data: BaselineDataLoader,
        validation_outcome_col_name: str,
        preprocessing_pipeline: PreprocessingPipeline,
        task: BaselineTask,
        logger: BaselineLogger,
    ):
        self.training_data = training_data.load()
        self.training_outcome_col_name = training_outcome_col_name
        self.validation_data = validation_data.load()
        self.validation_outcome_col_name = validation_outcome_col_name
        self.preprocessing_pipeline = preprocessing_pipeline
        self.problem_type = task
        self.logger = logger

    def train(self) -> TrainingResult:
        training_data_preprocessed = self.preprocessing_pipeline.apply(
            data=self.training_data,
        )
        validation_data = self.preprocessing_pipeline.apply(
            data=self.validation_data,
        )

        self.problem_type.train(
            x=training_data_preprocessed.drop(self.training_outcome_col_name),
            y=training_data_preprocessed.select(self.training_outcome_col_name),
        )

        result = self.problem_type.evaluate(
            x=validation_data.drop(self.validation_outcome_col_name),
            y=validation_data.select(self.validation_outcome_col_name),
        )

        self.logger.log_metric(result.metric)

        return result
