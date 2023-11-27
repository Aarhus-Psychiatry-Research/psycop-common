# ruff: noqa


def populate_baseline_registry() -> None:
    """
    Populate the registry with all the registered functions

    It is also possible to do this using hooks, but this is more explicit
    and easier to debug for people who are not familiar with python setup hooks.
    """
    # Loggers
    # Suggesters
    from ..hyperparameter_suggester.hyperparameter_suggester import SuggesterSpace
    from ..loggers.terminal_logger import TerminalLogger

    # Trainers
    from ..trainer.cross_validator_trainer import CrossValidatorTrainer

    # Data filters
    from ..trainer.data.data_filters.geography import GeographyDataFilter
    from ..trainer.data.data_filters.original_ids import OriginalIDDataFilter

    # Data
    from ..trainer.data.dataloaders import (
        FilteredDataLoader,
        ParquetVerticalConcatenator,
    )
    from ..trainer.data.minimal_test_data import MinimalTestData

    # Preprocessing
    from ..trainer.preprocessing.pipeline import BaselinePreprocessingPipeline
    from ..trainer.preprocessing.steps.cell_transformers import BoolToInt
    from ..trainer.preprocessing.steps.column_filters import (
        FilterColumnsWithinSubset,
        RegexColumnBlacklist,
        TemporalColumnFilter,
    )
    from ..trainer.preprocessing.steps.column_validator import (
        ColumnExistsValidator,
        ColumnPrefixExpectation,
    )
    from ..trainer.preprocessing.steps.row_filters import AgeFilter
    from ..trainer.split_trainer import SplitTrainer
    from ..trainer.task.binary_classification.binary_classification_pipeline import (
        BinaryClassificationPipeline,
    )
    from ..trainer.task.binary_classification.binary_classification_task import (
        BinaryClassificationTask,
    )
    from ..trainer.task.binary_classification.binary_metrics.binary_auroc import (
        BinaryAUROC,
    )

    # Estimator steps
    from ..trainer.task.estimator_steps.logistic_regression import (
        logistic_regression_step,
    )
    from ..trainer.task.estimator_steps.xgboost import xgboost_classifier_step

    # Tasks
    from ..trainer.task.pipeline_constructor import pipeline_constructor
