from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import polars as pl

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader
from psycop.common.model_training_v2.trainer.base_trainer import BaselineTrainer, TrainingResult
from psycop.common.model_training_v2.trainer.preprocessing.pipeline import PreprocessingPipeline
from psycop.common.model_training_v2.trainer.task.base_metric import BaselineMetric
from psycop.common.model_training_v2.trainer.task.base_task import BaselineTask


@BaselineRegistry.trainers.register("split_trainer_separate_preprocessing")
@dataclass
class SplitTrainerSeparatePreprocessing(BaselineTrainer):
    uuid_col_name: str
    group_col_name: str
    training_data: BaselineDataLoader
    training_outcome_col_name: str
    validation_data: BaselineDataLoader
    validation_outcome_col_name: str
    training_preprocessing_pipeline: PreprocessingPipeline
    validation_preprocessing_pipeline: PreprocessingPipeline
    task: BaselineTask
    metric: BaselineMetric
    additional_metrics: list[BaselineMetric] | None = None

    # When using sklearn pipelines, the outcome column must retain its name
    # throughout the pipeline.
    # To accomplish this, we rename the two outcomes to a shared name.
    _shared_outcome_col_name = "outcome"

    @property
    def outcome_columns(self) -> Sequence[str]:
        return [self.training_outcome_col_name, self.validation_outcome_col_name]

    @property
    def non_predictor_columns(self) -> Sequence[str]:
        return [self.uuid_col_name, self.group_col_name, *self.outcome_columns]

    def train(self) -> TrainingResult:
        training_data_preprocessed = self.training_preprocessing_pipeline.apply(
            data=self.training_data.load()
        )
        validation_data_preprocessed = self.validation_preprocessing_pipeline.apply(
            data=self.validation_data.load()
        )

        x = training_data_preprocessed.drop(self.non_predictor_columns, axis=1)
        self.logger.info(f"The model sees these predictors:\n\t{x.columns}")

        training_y = training_data_preprocessed[self.training_outcome_col_name]
        self.logger.info(f"\tOutcome: {self.training_outcome_col_name}")

        self.task.train(x=x, y=pd.DataFrame(training_y), y_col_name=self.training_outcome_col_name)

        y_hat_prob = self.task.predict_proba(
            x=validation_data_preprocessed.drop(self.non_predictor_columns, axis=1)
        )
        main_metric = self.metric.calculate(
            y=validation_data_preprocessed[self.validation_outcome_col_name], y_hat_prob=y_hat_prob
        )
        self._log_main_metric(main_metric)
        self._log_sklearn_pipe()

        if self.additional_metrics:
            for metric in self.additional_metrics:
                additional_metric = metric.calculate(
                    y=validation_data_preprocessed[self.validation_outcome_col_name],
                    y_hat_prob=y_hat_prob,
                )
                self.logger.log_metric(additional_metric)

        eval_df = pl.DataFrame(
            {
                "y": validation_data_preprocessed[self.validation_outcome_col_name],
                "y_hat_prob": y_hat_prob,
                "pred_time_uuid": validation_data_preprocessed[self.uuid_col_name],
            }
        )
        self.logger.log_dataset(dataframe=eval_df, filename="eval_df.parquet")

        return TrainingResult(metric=main_metric, df=eval_df)


@BaselineRegistry.trainers.register("split_trainer")
@dataclass
class SplitTrainer(BaselineTrainer):
    uuid_col_name: str
    group_col_name: str
    training_data: BaselineDataLoader
    training_outcome_col_name: str
    validation_data: BaselineDataLoader
    validation_outcome_col_name: str
    preprocessing_pipeline: PreprocessingPipeline
    task: BaselineTask
    metric: BaselineMetric
    additional_metrics: list[BaselineMetric] | None = None

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
        trainer = SplitTrainerSeparatePreprocessing(
            uuid_col_name=self.uuid_col_name,
            group_col_name=self.group_col_name,
            training_data=self.training_data,
            training_outcome_col_name=self.training_outcome_col_name,
            validation_data=self.validation_data,
            validation_outcome_col_name=self.validation_outcome_col_name,
            training_preprocessing_pipeline=self.preprocessing_pipeline,
            validation_preprocessing_pipeline=self.preprocessing_pipeline,
            task=self.task,
            metric=self.metric,
            additional_metrics=self.additional_metrics,
        )
        trainer.set_logger(self.logger)
        return trainer.train()
