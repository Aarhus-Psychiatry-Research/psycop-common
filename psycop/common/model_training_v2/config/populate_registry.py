# ruff: noqa


def populate_baseline_registry() -> None:
    """
    Populate the registry with all the registered functions

    It is also possible to do this using hooks, but this is more explicit
    and easier to debug for people who are not familiar with python setup hooks.
    """
    # Suggesters
    from ..hyperparameter_suggester.hyperparameter_suggester import SuggesterSpace

    # Loggers
    from ..loggers.terminal_logger import TerminalLogger
    from ..loggers.disk_logger import DiskLogger
    from ..loggers.multi_logger import MultiLogger
    from ..loggers.mlflow_logger import MLFlowLogger

    # Trainers
    from ..trainer.cross_validator_trainer import CrossValidatorTrainer

    # Data filters
    from ..trainer.data.data_filters.geography import RegionalFilter
    from ..trainer.data.data_filters.original_ids import FilterByEntityID

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

    # Suggesters
    from ..hyperparameter_suggester.hyperparameter_suggester import SuggesterSpace

    # Data
    from ..trainer.data.dataloaders import (
        ParquetVerticalConcatenator,
        FilteredDataLoader,
    )
    from ..trainer.data.minimal_test_data import MinimalTestData
    from ..trainer.data.data_filters.minimal_data_filter_test_data import (
        mock_regional_move_df,
        mock_split_id_sequence,
    )
    from ..trainer.task.binary_classification.binary_classification_task import (
        BinaryClassificationTask,
    )

    # Data filters
    from ..trainer.data.data_filters.geography import RegionalFilter
