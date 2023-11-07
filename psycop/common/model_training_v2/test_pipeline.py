from pathlib import Path

from psycop.common.model_training_v2.classifier_pipelines.binary_classification_pipeline import (
    BinaryClassificationPipeline,
)
from psycop.common.model_training_v2.classifier_pipelines.estimator_steps.logistic_regression import (
    logistic_regression_step,
)
from psycop.common.model_training_v2.loggers.base_logger import (
    TerminalLogger,
)
from psycop.common.model_training_v2.metrics.binary_metrics.binary_auroc import (
    BinaryAUROC,
)
from psycop.common.model_training_v2.pipeline import (
    BaselineSchema,
    train_baseline_model,
)
from psycop.common.model_training_v2.presplit_preprocessing.pipeline import (
    BaselinePreprocessingPipeline,
)
from psycop.common.model_training_v2.presplit_preprocessing.presplit_steps.filters import (
    AgeFilter,
)
from psycop.common.model_training_v2.problem_type.binary_classification import (
    BinaryClassification,
)
from psycop.common.model_training_v2.training_method.split_trainer import (
    SplitTrainer,
)
from psycop.common.test_utils.str_to_df import str_to_pl_df


def test_v2_train_model_pipeline(tmpdir: Path):
    training_data = str_to_pl_df(
        """pred_time_uuid,pred_1,outcome,pred_age
                                     1,1,1,1
                                     2,1,1,99
                                     3,1,1,99
                                     4,0,0,99
                                     5,0,0,99
                                     6,0,0,99
                                     """,
    )
    schema = BaselineSchema(
        experiment_path=tmpdir,
        logger=TerminalLogger(),
        training_method=SplitTrainer(
            training_data=training_data,
            training_outcome_col_name="outcome",
            validation_data=training_data,
            validation_outcome_col_name="outcome",
            preprocessing_pipeline=BaselinePreprocessingPipeline(
                [AgeFilter(min_age=4, max_age=99, age_col_name="pred_age")],
            ),
            problem_type=BinaryClassification(
                pipe=BinaryClassificationPipeline(steps=[logistic_regression_step()]),
                main_metric=BinaryAUROC(),
            ),
            logger=logger,
        ),
    )

    assert train_baseline_model(schema) == 1.0
