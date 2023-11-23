# ruff: noqa


def populate_baseline_registry() -> None:
    """
    Populate the registry with all the registered functions

    It is also possible to do this using hooks, but this is more explicit
    and easier to debug for people who are not familiar with python setup hooks.
    """
    # Loggers
    from ..loggers.terminal_logger import TerminalLogger

    # Preprocessing
    from ..trainer.preprocessing.pipeline import BaselinePreprocessingPipeline
    from ..trainer.preprocessing.steps.row_filters import AgeFilter
    from ..trainer.preprocessing.steps.column_validator import (
        ColumnExistsValidator,
        ColumnPrefixExpectation,
    )
    from ..trainer.preprocessing.steps.column_filters import (
        TemporalColumnFilter,
        RegexColumnBlacklist,
    )

    # Trainers
    from ..trainer.cross_validator_trainer import CrossValidatorTrainer
    from ..trainer.split_trainer import SplitTrainer

    # Tasks
    from ..trainer.task.pipeline_constructor import pipeline_constructor
    from ..trainer.task.binary_classification.binary_classification_pipeline import (
        BinaryClassificationPipeline,
    )
    from ..trainer.task.binary_classification.binary_metrics.binary_auroc import (
        BinaryAUROC,
    )

    # Estimator steps
    from ..trainer.task.estimator_steps.logistic_regression import (
        logistic_regression_step,
    )
    from ..trainer.task.estimator_steps.xgboost import (
        xgboost_classifier_step,
    )

    # Suggesters
    from ..hyperparameter_suggester.hyperparameter_suggester import SuggesterSpace

    # Data
    from ..trainer.data.minimal_test_data import MinimalTestData
    from ..trainer.task.binary_classification.binary_classification_task import (
        BinaryClassificationTask,
    )
