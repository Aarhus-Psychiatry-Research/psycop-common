from collections.abc import Iterable, Sequence

import pandas as pd
import polars as pl

from psycop.common.model_training_v2.artifact_savers.base_artifact_saver import (
    BaselineArtifactSaver,
)
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
class SplitTrainer(BaselineTrainer):
    def __init__(
        self,
        training_data: BaselineDataLoader,
        training_outcome_col_name: str,
        validation_data: BaselineDataLoader,
        validation_outcome_col_name: str,
        preprocessing_pipeline: PreprocessingPipeline,
        task: BaselineTask,
        metric: BaselineMetric,
        logger: BaselineLogger,
        artifact_savers: Iterable[BaselineArtifactSaver] | None = None,
    ):
        self.training_data = training_data.load()
        self.training_outcome_col_name = training_outcome_col_name
        self.validation_data = validation_data.load()
        self.validation_outcome_col_name = validation_outcome_col_name
        self.preprocessing_pipeline = preprocessing_pipeline
        self.task = task
        self.metric = metric
        self.logger = logger
        self.artifact_savers = artifact_savers

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

        x = training_data_preprocessed.drop(self.outcome_columns, axis=1)
        self.logger.info(
            f"Training on:\n\tFeatures: {training_data_preprocessed.columns}",
        )
        self.logger.info(f"\tOutcome: {self.shared_outcome_col_name}")

        training_y = training_data_preprocessed[self.training_outcome_col_name]
        self.task.train(
            x=x,
            y=pd.DataFrame(training_y),
            y_col_name=self.training_outcome_col_name,
        )

        y_hat_prob = self.task.predict_proba(
            x=validation_data_preprocessed.drop(self.outcome_columns, axis=1),
        )

        main_metric = self.metric.calculate(y=training_y, y_hat_prob=y_hat_prob)
        training_result = TrainingResult(
            metric=main_metric,
            df=pl.DataFrame(
                pd.concat([validation_data_preprocessed, y_hat_prob], axis=1),
            ),
        )

        self.logger.log_metric(main_metric)
        self._save_artifacts(training_result=training_result)

        return training_result

    def _save_artifacts(self, training_result: TrainingResult) -> None:
        if self.artifact_savers is not None:
            for artifact_saver in self.artifact_savers:
                artifact_saver.save_artifact(
                    trainer=self,
                    training_result=training_result,
                )
