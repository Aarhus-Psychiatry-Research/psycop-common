from dataclasses import dataclass

import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.base_dataloader import BaselineDataLoader
from psycop.common.model_training_v2.trainer.base_trainer import BaselineTrainer, TrainingResult
from psycop.common.model_training_v2.trainer.preprocessing.pipeline import PreprocessingPipeline
from psycop.common.model_training_v2.trainer.task.base_metric import (
    BaselineMetric,
    CalculatedMetric,
)
from psycop.common.model_training_v2.trainer.task.base_task import BaselineTask


@BaselineRegistry.trainers.register("selective_crossval_trainer")
@dataclass
class SelectiveCrossValidatorTrainer(BaselineTrainer):
    uuid_col_name: str
    outcome_col_name: str
    training_data: BaselineDataLoader
    additional_data: BaselineDataLoader
    preprocessing_pipeline: PreprocessingPipeline
    task: BaselineTask
    metric: BaselineMetric
    additional_metrics: list[BaselineMetric] | None = None
    n_splits: int = 5
    group_col_name: str = "dw_ek_borger"
    """A cross-validator trainer that trains on a combination of `training_data`
    and `additional_data` and evaluates on the `training_data` only. This can be
    useful for training with synthetic data and only evluating on the real data
    or for training with prevalent cases and evaluating on incident cases only."""

    def train(self) -> TrainingResult:
        # assign a dummy variable "_do_eval" do indicate whether the data point should be used for evaluation
        training_data_preprocessed = self.preprocessing_pipeline.apply(
            data=self.training_data.load()
        ).assign(_do_eval=1)
        additional_data_preprocessed = self.preprocessing_pipeline.apply(
            data=self.additional_data.load()
        ).assign(_do_eval=0)

        training_data_preprocessed = pd.concat(
            [training_data_preprocessed, additional_data_preprocessed], axis=0
        ).reset_index(drop=True)

        training_data_preprocessed["dw_ek_borger"] = training_data_preprocessed[
            "dw_ek_borger"
        ].astype(str)

        X = training_data_preprocessed.drop([self.outcome_col_name, self.uuid_col_name], axis=1)
        y = pd.DataFrame(
            training_data_preprocessed[self.outcome_col_name], columns=[self.outcome_col_name]
        )
        self.logger.info(f"\tOutcome: {list(y.columns)}")

        folds = StratifiedGroupKFold(n_splits=self.n_splits).split(
            X=X, y=y, groups=training_data_preprocessed[self.group_col_name]
        )

        training_data_for_eval = training_data_preprocessed.loc[
            training_data_preprocessed["_do_eval"] == 1
        ].copy()

        for i, (train_idxs, val_idxs) in enumerate(folds):
            X_train, y_train = (X.loc[train_idxs], y.loc[train_idxs])

            X_train = X_train.drop(columns=[self.group_col_name, "_do_eval"])

            X_val, y_val = (X.loc[val_idxs], y.loc[val_idxs])

            # only validate on training_data, not additional data
            X_val = X_val[X_val["_do_eval"] == 1].drop(columns=[self.group_col_name, "_do_eval"])
            y_val = y_val[y_val.index.isin(X_val.index)]

            self.task.train(X_train, y_train, y_col_name=self.outcome_col_name)

            y_hat_prob = self.task.predict_proba(X_train)

            within_fold_metric = self.metric.calculate(
                y=y_train[self.outcome_col_name],
                y_hat_prob=y_hat_prob,
                name_prefix=f"within_fold_{i}",
            )
            self.logger.log_metric(within_fold_metric)
            self.logger.log_metric(CalculatedMetric(name=f"within_fold_{i}_n", value=len(y_train)))

            oof_y_hat_prob = self.task.predict_proba(X_val)

            oof_metric = self.metric.calculate(
                y=y_val[self.outcome_col_name],
                y_hat_prob=oof_y_hat_prob,
                name_prefix=f"out_of_fold_{i}",
            )
            self.logger.log_metric(oof_metric)

            training_data_for_eval.loc[y_val.index, "oof_y_hat_prob"] = oof_y_hat_prob.to_list()  # type: ignore

        main_metric = self.metric.calculate(
            y=training_data_for_eval[self.outcome_col_name],
            y_hat_prob=training_data_for_eval["oof_y_hat_prob"],
            name_prefix="all_oof",
        )
        self._log_main_metric(main_metric)
        self._log_sklearn_pipe()

        if self.additional_metrics:
            for metric in self.additional_metrics:
                additional_metric = metric.calculate(
                    y=training_data_for_eval[self.outcome_col_name],
                    y_hat_prob=training_data_for_eval["oof_y_hat_prob"],
                    name_prefix="all_oof",
                )
                self.logger.log_metric(additional_metric)

        eval_df = pl.DataFrame(
            {
                "y": training_data_for_eval[self.outcome_col_name],
                "y_hat_prob": training_data_for_eval["oof_y_hat_prob"],
                "pred_time_uuid": training_data_for_eval[self.uuid_col_name],
            }
        )
        self.logger.log_dataset(dataframe=eval_df, filename="eval_df.parquet")

        return TrainingResult(metric=main_metric, df=eval_df)
