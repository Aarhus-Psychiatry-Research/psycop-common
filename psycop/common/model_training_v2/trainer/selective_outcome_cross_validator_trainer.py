from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader
from psycop.common.model_training_v2.trainer.base_trainer import BaselineTrainer, TrainingResult
from psycop.common.model_training_v2.trainer.preprocessing.pipeline import PreprocessingPipeline
from psycop.common.model_training_v2.trainer.task.base_metric import BaselineMetric
from psycop.common.model_training_v2.trainer.task.base_task import BaselineTask


@BaselineRegistry.trainers.register("selective_outcome_crossval_trainer")
@dataclass
class SelectiveOutcomeCrossValidatorTrainer(BaselineTrainer):
    uuid_col_name: str
    group_col_name: str
    training_data: BaselineDataLoader
    preprocessing_pipeline: PreprocessingPipeline
    training_outcome_col_name: str
    validation_outcome_col_name: str
    task: BaselineTask
    metric: BaselineMetric
    n_splits: int = 5

    """A cross-validator trainer that trains on a specific outcome column and evaluates on another outcome column."""

    # When using sklearn pipelines, the outcome column must retain its name
    # throughout the pipeline.
    # To accomplish this, we rename the two outcomes to a shared name.
    _shared_outcome_col_name = "outcome"

    @property
    def non_predictor_columns(self) -> Sequence[str]:
        return [self.uuid_col_name, self.group_col_name]

    def train(self) -> TrainingResult:
        training_data_preprocessed = self.preprocessing_pipeline.apply(
            data=self.training_data.load()
        )

        X = training_data_preprocessed.drop(self.non_predictor_columns, axis=1)
        self.logger.info(f"The model sees these predictors:\n\t{X.columns}")

        training_y = pd.DataFrame(
            training_data_preprocessed[self.training_outcome_col_name],
            columns=[self.training_outcome_col_name],
        )

        self.logger.info(f"\tOutcome: {self.training_outcome_col_name}")

        folds = StratifiedGroupKFold(n_splits=self.n_splits).split(
            X=X, y=training_y, groups=training_data_preprocessed[self.group_col_name]
        )

        for i, (train_idxs, val_idxs) in enumerate(folds):
            X_train, y_train = (X.loc[train_idxs], training_y.loc[train_idxs])

            X_val, y_val = (
                X.loc[val_idxs],
                pd.Series(X[self.validation_outcome_col_name][val_idxs]),
            )

            X_train = X_train.drop(
                [self.training_outcome_col_name, self.validation_outcome_col_name], axis=1
            )
            X_val = X_val.drop(
                [self.training_outcome_col_name, self.validation_outcome_col_name], axis=1
            )

            self.task.train(X_train, y_train, y_col_name=self.training_outcome_col_name)

            y_hat_prob = self.task.predict_proba(X_train)

            within_fold_metric = self.metric.calculate(
                y=y_train[self.training_outcome_col_name],
                y_hat_prob=y_hat_prob,
                name_prefix=f"within_fold_{i}",
            )
            self.logger.log_metric(within_fold_metric)

            oof_y_hat_prob = self.task.predict_proba(X_val)

            oof_metric = self.metric.calculate(
                y=y_val, y_hat_prob=oof_y_hat_prob, name_prefix=f"out_of_fold_{i}"
            )
            self.logger.log_metric(oof_metric)

            training_data_preprocessed.loc[val_idxs, "y_val"] = y_val

            training_data_preprocessed.loc[val_idxs, "oof_y_hat_prob"] = oof_y_hat_prob.to_list()  # type: ignore

        main_metric = self.metric.calculate(
            y=training_data_preprocessed["y_val"],
            y_hat_prob=training_data_preprocessed["oof_y_hat_prob"],
            name_prefix="all_oof",
        )
        self._log_main_metric(main_metric)
        self._log_sklearn_pipe()

        eval_df = pl.DataFrame(
            {
                "y": training_data_preprocessed["y_val"],
                "y_hat_prob": training_data_preprocessed["oof_y_hat_prob"],
                "pred_time_uuid": training_data_preprocessed[self.uuid_col_name],
            }
        )
        self.logger.log_dataset(dataframe=eval_df, filename="eval_df.parquet")

        return TrainingResult(metric=main_metric, df=eval_df)
