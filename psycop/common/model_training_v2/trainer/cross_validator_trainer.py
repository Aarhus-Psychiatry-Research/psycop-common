import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold

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
from psycop.common.model_training_v2.trainer.task.base_metric import BaselineMetric
from psycop.common.model_training_v2.trainer.task.base_task import BaselineTask


@BaselineRegistry.trainers.register("crossval_trainer")
class CrossValidatorTrainer(BaselineTrainer):
    def __init__(
        self,
        training_data: BaselineDataLoader,
        outcome_col_name: str,
        preprocessing_pipeline: PreprocessingPipeline,
        task: BaselineTask,
        metric: BaselineMetric,
        logger: BaselineLogger,
        n_splits: int = 5,
        group_col_name: str = "dw_ek_borger",
    ):
        self.training_data = training_data.load()
        self.outcome_col_name = outcome_col_name
        self.preprocessing_pipeline = preprocessing_pipeline
        self.task = task
        self.metric = metric
        self.n_splits = n_splits
        self.group_col_name = group_col_name
        self.logger = logger

    def train(self) -> TrainingResult:
        training_data_preprocessed = self.preprocessing_pipeline.apply(
            data=self.training_data,
        )

        X = training_data_preprocessed.drop(
            self.outcome_col_name,
            axis=1,
        )
        y = pd.DataFrame(
            training_data_preprocessed[self.outcome_col_name],
            columns=[self.outcome_col_name],
        )

        folds = StratifiedGroupKFold(n_splits=self.n_splits).split(
            X=X,
            y=y,
            groups=training_data_preprocessed[self.group_col_name],
        )

        for i, (train_idxs, val_idxs) in enumerate(folds):
            X_train, y_train = (
                X.loc[train_idxs],
                y.loc[train_idxs],
            )

            self.task.train(X_train, y_train, y_col_name=self.outcome_col_name)

            y_hat_prob = self.task.predict_proba(
                X_train,
            )

            within_fold_metric = self.metric.calculate(
                y=y_train[self.outcome_col_name],
                y_hat_prob=y_hat_prob,
            )

            metric_name = within_fold_metric.name
            within_fold_metric.name = f"within_fold_{metric_name}_{i}"
            self.logger.log_metric(
                within_fold_metric,
            )

            oof_y_hat_prob = self.task.predict_proba(
                X.loc[val_idxs],
            )

            oof_metric = self.metric.calculate(
                y=y.loc[val_idxs][self.outcome_col_name],
                y_hat_prob=oof_y_hat_prob,
            )

            oof_metric.name = f"oof_{metric_name}_{i}"
            self.logger.log_metric(
                oof_metric,
            )

            training_data_preprocessed.loc[
                val_idxs,
                "oof_y_hat_prob",
            ] = oof_y_hat_prob.to_list()  # type: ignore

        main_metric = self.metric.calculate(
            y=training_data_preprocessed[self.outcome_col_name],
            y_hat_prob=training_data_preprocessed["oof_y_hat_prob"],
        )
        self.logger.log_metric(main_metric)

        return TrainingResult(
            metric=main_metric,
            df=pl.DataFrame(training_data_preprocessed),
        )
