from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl
from numpy import ndarray
from sklearn.model_selection import StratifiedGroupKFold

from psycop.common.global_utils.cache import shared_cache
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader
from psycop.common.model_training_v2.trainer.base_trainer import BaselineTrainer, TrainingResult
from psycop.common.model_training_v2.trainer.preprocessing.pipeline import PreprocessingPipeline
from psycop.common.model_training_v2.trainer.task.base_metric import BaselineMetric
from psycop.common.model_training_v2.trainer.task.base_task import BaselineTask


@shared_cache().cache()
def cached_folds(
    n_splits: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: pd.Series,  # type: ignore
) -> list[tuple[ndarray[Any, Any], ndarray[Any, Any]]]:
    return list(StratifiedGroupKFold(n_splits=n_splits).split(X=X, y=y, groups=groups))


@BaselineRegistry.trainers.register("crossval_trainer")
@dataclass
class CrossValidatorTrainer(BaselineTrainer):
    uuid_col_name: str
    outcome_col_name: str
    training_data: BaselineDataLoader
    preprocessing_pipeline: PreprocessingPipeline
    task: BaselineTask
    metric: BaselineMetric
    additional_metrics: BaselineMetric | None = None
    n_splits: int = 5
    group_col_name: str = "dw_ek_borger"

    def train(self) -> TrainingResult:
        training_data_preprocessed = self.preprocessing_pipeline.apply(
            data=self.training_data.load()
        )

        X = training_data_preprocessed.drop([self.outcome_col_name, self.uuid_col_name], axis=1)
        y = pd.DataFrame(
            training_data_preprocessed[self.outcome_col_name], columns=[self.outcome_col_name]
        )
        self.logger.info(f"\tOutcome: {list(y.columns)}")

        folds = cached_folds(
            n_splits=self.n_splits, X=X, y=y, groups=training_data_preprocessed[self.group_col_name]
        )

        for i, (train_idxs, val_idxs) in enumerate(folds):
            X_train, y_train = (X.loc[train_idxs], y.loc[train_idxs])

            X_train = X_train.drop(columns=self.group_col_name)
            self.task.train(X_train, y_train, y_col_name=self.outcome_col_name)

            y_hat_prob = self.task.predict_proba(X_train)

            within_fold_metric = self.metric.calculate(
                y=y_train[self.outcome_col_name],
                y_hat_prob=y_hat_prob,
                name_prefix=f"within_fold_{i}",
            )
            self.logger.log_metric(within_fold_metric)

            oof_y_hat_prob = self.task.predict_proba(
                X.loc[val_idxs].drop(columns=self.group_col_name)
            )

            oof_metric = self.metric.calculate(
                y=y.loc[val_idxs][self.outcome_col_name],
                y_hat_prob=oof_y_hat_prob,
                name_prefix=f"out_of_fold_{i}",
            )
            self.logger.log_metric(oof_metric)

            training_data_preprocessed.loc[val_idxs, "oof_y_hat_prob"] = oof_y_hat_prob.to_list()  # type: ignore

        main_metric = self.metric.calculate(
            y=training_data_preprocessed[self.outcome_col_name],
            y_hat_prob=training_data_preprocessed["oof_y_hat_prob"],
            name_prefix="all_oof",
        )
        self._log_main_metric(main_metric)
        self._log_sklearn_pipe()

        if self.additional_metrics:
            additional_metric = self.additional_metrics.calculate(
                y=training_data_preprocessed[self.outcome_col_name],
                y_hat_prob=training_data_preprocessed["oof_y_hat_prob"],
                name_prefix="all_oof",
            )
            self.logger.log_metric(additional_metric)

        eval_df = pl.DataFrame(
            {
                "y": training_data_preprocessed[self.outcome_col_name],
                "y_hat_prob": training_data_preprocessed["oof_y_hat_prob"],
                "pred_time_uuid": training_data_preprocessed[self.uuid_col_name],
            }
        )
        self.logger.log_dataset(dataframe=eval_df, filename="eval_df.parquet")

        return TrainingResult(metric=main_metric, df=eval_df)
