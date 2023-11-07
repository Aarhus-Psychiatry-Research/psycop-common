from pathlib import Path

import polars as pl

from psycop.common.model_training_v2.classifier_pipelines.binary_classification_pipeline import (
    BinaryClassificationPipeline,
)
from psycop.common.model_training_v2.classifier_pipelines.estimator_steps.xgboost import (
    xgboost_classifier_step,
)
from psycop.common.model_training_v2.loggers.base_logger import (
    BaselineLogger,
    TerminalLogger,
)
from psycop.common.model_training_v2.metrics.binary_metrics.binary_auroc import (
    BinaryAUROC,
)
from psycop.common.model_training_v2.pipeline import BaselineSchema
from psycop.common.model_training_v2.presplit_preprocessing.pipeline import (
    BaselinePreprocessingPipeline,
)
from psycop.common.model_training_v2.presplit_preprocessing.presplit_steps.filters import (
    AgeFilter,
)
from psycop.common.model_training_v2.problem_type.problem_type_base import (
    BinaryClassification,
)
from psycop.common.model_training_v2.training_method.base_training_method import (
    SplitTrainer,
)
from psycop.common.test_utils.str_to_df import str_to_pl_df


def test_v2_train_model_pipeline():
    training_data = str_to_pl_df("""pred_time_uuid,pred_1,pred_age,outcome
                                     1,1,1,1
                                     2,1,1,99
                                     3,1,1,99
                                     4,0,0,99
                                     5,0,0,99
                                     6,0,0,99
                                     """)
    logger = TerminalLogger()
    
    cfg = BaselineSchema(
        experiment_path=Path(""),
        logger=logger,
        training_method=SplitTrainer(
            training_data=training_data,
            training_outcome_col_name="outcome",
            validation_data=training_data,
            validation_outcome_col_name="outcome",
            preprocessing_pipeline=BaselinePreprocessingPipeline([AgeFilter(min_age=4, max_age=99, age_col_name="pred_age")]),
            problem_type=BinaryClassification(pipe=BinaryClassificationPipeline(steps=[xgboost_classifier_step()]), main_metric=BinaryAUROC()),
            logger=logger,
        ),
    )