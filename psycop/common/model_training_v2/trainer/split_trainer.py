from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import polars as pl

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
from psycop.common.model_training_v2.trainer.task.base_metric import BaselineMetric
from psycop.common.model_training_v2.trainer.task.base_task import (
    BaselineTask,
)


@BaselineRegistry.trainers.register("split_trainer")
@dataclass(frozen=True)
class SplitTrainer(BaselineTrainer):
    uuid_col_name: str
    training_data: BaselineDataLoader
    training_outcome_col_name: str
    validation_data: BaselineDataLoader
    validation_outcome_col_name: str
    preprocessing_pipeline: PreprocessingPipeline
    task: BaselineTask
    metric: BaselineMetric
    logger: BaselineLogger

    # When using sklearn pipelines, the outcome column must retain its name
    # throughout the pipeline.
    # To accomplish this, we rename the two outcomes to a shared name.
    _shared_outcome_col_name = "outcome"

    @property
    def outcome_columns(self) -> Sequence[str]:
        return [self.training_outcome_col_name, self.validation_outcome_col_name]

    @property
    def non_predictor_columns(self) -> Sequence[str]:
        return [self.uuid_col_name, *self.outcome_columns]

    def train(self) -> TrainingResult:
        training_data_preprocessed = self.preprocessing_pipeline.apply(
            data=self.training_data.load(),
        )
        validation_data_preprocessed = self.preprocessing_pipeline.apply(
            data=self.validation_data.load(),
        )

        x = training_data_preprocessed.drop(self.non_predictor_columns, axis=1)
        self.logger.info(
            f"Training on:\n\tFeatures: {training_data_preprocessed.columns}",
        )

        training_y = training_data_preprocessed[self.training_outcome_col_name]
        self.logger.info(f"\tOutcome: {self.training_outcome_col_name}")

        self.task.train(
            x=x,
            y=pd.DataFrame(training_y),
            y_col_name=self.training_outcome_col_name,
        )

        y_hat_prob = self.task.predict_proba(
            x=validation_data_preprocessed.drop(self.non_predictor_columns, axis=1),
        )

        main_metric = self.metric.calculate(y=training_y, y_hat_prob=y_hat_prob)
        self._log_main_metric(main_metric)
        self._log_sklearn_pipe()

        return TrainingResult(
            metric=main_metric,
            df=pl.DataFrame(
                pd.concat([validation_data_preprocessed, y_hat_prob], axis=1),
            ),
        )
