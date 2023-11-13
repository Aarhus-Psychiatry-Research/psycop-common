import pandas as pd
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
from psycop.common.model_training_v2.trainer.task.base_task import BaselineTask


@BaselineRegistry.trainers.register("crossval_trainer")
class CrossValidatorTrainer(BaselineTrainer):
    def __init__(
        self,
        training_data: BaselineDataLoader,  # make into list?
        training_outcome_col_name: str,
        preprocessing_pipeline: PreprocessingPipeline,
        task: BaselineTask,
        logger: BaselineLogger,
        n_splits: int = 5,
        group_col_name: str = "dw_ek_borger",
    ):
        self.training_data = training_data.load()
        self.training_outcome_col_name = training_outcome_col_name
        self.preprocessing_pipeline = preprocessing_pipeline
        self.task = task
        self.n_splits = n_splits
        self.group_col_name = group_col_name
        self.logger = logger

    def train(self) -> TrainingResult:
        training_data_preprocessed = self.preprocessing_pipeline.apply(
            data=self.training_data,
        )

        X = training_data_preprocessed.drop(
            self.training_outcome_col_name,
            axis=1,
        )
        y = pd.DataFrame(
            training_data_preprocessed[self.training_outcome_col_name],
            columns=[self.training_outcome_col_name],
        )

        folds = StratifiedGroupKFold(n_splits=self.n_splits).split(
            X=X,
            y=y,
            groups=training_data_preprocessed[self.group_col_name],
        )

        training_data_preprocessed["y_hat_prob"] = 0

        for _i, (train_idxs, val_idxs) in enumerate(folds):
            X_train, y_train = (
                X.loc[train_idxs],
                y.loc[train_idxs],
            )

            self.task.train(X_train, y_train, y_col_name=self.training_outcome_col_name)

            y_pred = self.task.evaluate(
                X_train,
                y_train,
                y_col_name=self.training_outcome_col_name,
            )

            self.logger.log_metric(y_pred.metric)

            X_val, y_val = (
                X.loc[val_idxs],
                y.loc[val_idxs],
            )

            oof_y_pred = self.task.evaluate(
                X_val,
                y_val,
                y_col_name=self.training_outcome_col_name,
            )

            self.logger.log_metric(oof_y_pred.metric)

            oof_y_prob = self.task.predict_proba(
                X_val.drop(
                    self.group_col_name,
                    axis=1,
                ),
            )

            training_data_preprocessed.loc[val_idxs, "y_hat_prob"] = oof_y_prob

        result = self.task.evaluate(
            X,
            pd.DataFrame(training_data_preprocessed["y_hat_prob"]),
            y_col_name=self.training_outcome_col_name,
        )

        self.logger.log_metric(result.metric)

        return result
