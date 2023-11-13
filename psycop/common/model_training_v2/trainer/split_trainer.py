from collections.abc import Sequence

import pandas as pd

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
from psycop.common.model_training_v2.trainer.task.base_metric import BaseMetric
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
        metric: BaseMetric,
        logger: BaselineLogger,
    ):
        self.training_data = training_data.load()
        self.training_outcome_col_name = training_outcome_col_name
        self.validation_data = validation_data.load()
        self.validation_outcome_col_name = validation_outcome_col_name
        self.preprocessing_pipeline = preprocessing_pipeline
        self.task = task
        self.metric = metric
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

        training_y = training_data_preprocessed[self.training_outcome_col_name]
        self.task.train(
            x=training_data_preprocessed.drop(self.outcome_columns, axis=1),
            y=pd.DataFrame(training_y),
            y_col_name=self.training_outcome_col_name,
        )

        validation_data_preprocessed["y_hat"] = self.task.predict_proba(
            x=validation_data_preprocessed.drop(self.outcome_columns, axis=1),
        )
        eval_dataset = self.task.construct_eval_dataset(
            df=validation_data_preprocessed,
            y_hat_col="y_hat",
            y_col=self.validation_outcome_col_name,
        )

        main_metric = eval_dataset.calculate_metrics([self.metric])[0]
        self.logger.log_metric(main_metric)

        return TrainingResult(
            metric=main_metric,
            eval_dataset=eval_dataset,
        )
