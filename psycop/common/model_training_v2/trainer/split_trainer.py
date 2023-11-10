from collections.abc import Sequence

from psycop.common.model_training_v2.config.baseline_registry import (
    BaselineRegistry,
)
from psycop.common.model_training_v2.loggers.base_logger import (
    BaselineLogger,
)
from psycop.common.model_training_v2.trainer.base_dataloader import (
    BaselineDataLoader,
)
from psycop.common.model_training_v2.trainer.base_trainer import (
    BaselineTrainer,
    TrainingResult,
)
from psycop.common.model_training_v2.trainer.preprocessing.pipeline import (
    PreprocessingPipeline,
)
from psycop.common.model_training_v2.trainer.task.base_task import (
    BaselineTask,
)


@BaselineRegistry.trainers.register("split_trainer")
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

        # When using sklearn pipelines, the outcome column must retain its name
        # throughout the pipeline.
        # To accomplish this, we rename the two outcomes to a shared name.
        self.shared_outcome_col_name = "outcome"

    @property
    def outcome_columns(self) -> Sequence[str]:
        return [self.training_outcome_col_name, self.validation_outcome_col_name]

    def train(self) -> TrainingResult:
        training_data_preprocessed = self.preprocessing_pipeline.apply(
            data=self.training_data,
        )
        validation_data_preprocessed = self.preprocessing_pipeline.apply(
            data=self.validation_data,
        )

        training_y = training_data_preprocessed.select(
            self.training_outcome_col_name,
        ).rename({self.training_outcome_col_name: self.shared_outcome_col_name})

        validation_y = validation_data_preprocessed.select(
            self.validation_outcome_col_name,
        ).rename({self.validation_outcome_col_name: self.shared_outcome_col_name})

        self.problem_type.train(
            x=training_data_preprocessed.drop(self.outcome_columns),
            y=training_y,
        )
        result = self.problem_type.evaluate(
            x=validation_data_preprocessed.drop(self.outcome_columns),
            y=validation_y,
        )

        result = self._rename_result_col_name_to_validation_col_name(result)

        self.logger.log_metric(result.metric)

        return result

    def _rename_result_col_name_to_validation_col_name(self, result: TrainingResult) -> TrainingResult:
        result.eval_dataset.y = self.validation_outcome_col_name
        result.eval_dataset.df = result.eval_dataset.df.rename({self.shared_outcome_col_name: self.validation_outcome_col_name})
        return result
