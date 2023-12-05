import pickle as pkl

import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from psycop.common.model_training_v2.artifact_savers.save_task_to_disk import (
    TaskDiskSaver,
)
from psycop.common.model_training_v2.loggers.terminal_logger import TerminalLogger
from psycop.common.model_training_v2.trainer.base_trainer import TrainingResult
from psycop.common.model_training_v2.trainer.data.minimal_test_data import (
    MinimalTestData,
)
from psycop.common.model_training_v2.trainer.preprocessing.pipeline import (
    BaselinePreprocessingPipeline,
)
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filters import (
    AgeFilter,
)
from psycop.common.model_training_v2.trainer.split_trainer import SplitTrainer
from psycop.common.model_training_v2.trainer.task.base_metric import CalculatedMetric
from psycop.common.model_training_v2.trainer.task.binary_classification.binary_classification_pipeline import (
    BinaryClassificationPipeline,
)
from psycop.common.model_training_v2.trainer.task.binary_classification.binary_classification_task import (
    BinaryClassificationTask,
)
from psycop.common.model_training_v2.trainer.task.binary_classification.binary_metrics.binary_auroc import (
    BinaryAUROC,
)
from psycop.common.model_training_v2.trainer.task.estimator_steps.logistic_regression import (
    logistic_regression_step,
)


def test_task_disk_saver(tmpdir: str):
    trainer = SplitTrainer(
        training_data=MinimalTestData(),
        training_outcome_col_name="outcome",
        validation_data=MinimalTestData(),
        validation_outcome_col_name="outcome_val",
        preprocessing_pipeline=BaselinePreprocessingPipeline(
            AgeFilter(min_age=4, max_age=99, age_col_name="pred_age"),
        ),
        task=BinaryClassificationTask(
            pred_time_uuid_col_name="pred_time_uuid",
            task_pipe=BinaryClassificationPipeline(
                sklearn_pipe=Pipeline([logistic_regression_step()]),
            ),
        ),
        logger=TerminalLogger(),
        metric=BinaryAUROC(),
    )
    training_result = TrainingResult(
        metric=CalculatedMetric("mock_roc", value=0.5), df=pl.DataFrame({"x": [1]})
    )

    saver = TaskDiskSaver(experiment_path=tmpdir)
    saver.save_artifact(trainer=trainer, training_result=training_result)

    with saver.save_path.open("rb") as f:
        task = pkl.load(f)

    assert isinstance(task, BinaryClassificationTask)
