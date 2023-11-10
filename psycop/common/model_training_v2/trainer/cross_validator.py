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

        # do we need to convert to matrix? or would that be too slow?
        folds = StratifiedGroupKFold(n_splits=self.n_splits).split(
            X=training_data_preprocessed.drop(self.training_outcome_col_name),
            y=training_data_preprocessed.select(self.training_outcome_col_name),
            groups=training_data_preprocessed.select(self.group_col_name),
        )

        for _i, (train_idxs, val_idxs) in enumerate(folds):
            X_train, y_train = (
                training_data_preprocessed.loc[train_idxs],
                training_data_preprocessed.loc[train_idxs],
            )
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict_proba(X_train)[:, 1]

            roc_auc_score(y_train, y_pred)

            oof_y_pred = pipe.predict_proba(X.loc[val_idxs])[  # type: ignore
                :,
                1,
            ]

            roc_auc_score(y.loc[val_idxs], oof_y_pred)

            train_df.loc[val_idxs, "oof_y_hat"] = oof_y_pred

        self.task.train(
            x=training_data_preprocessed.drop(self.training_outcome_col_name),
            y=training_data_preprocessed.select(self.training_outcome_col_name),
        )
        # cross-validation
        result = self.task.evaluate(
            x=validation_data_preprocessed.drop(self.validation_outcome_col_name),
            y=validation_data_preprocessed.select(self.validation_outcome_col_name),
        )

        self.logger.log_metric(result.metric)

        return result
