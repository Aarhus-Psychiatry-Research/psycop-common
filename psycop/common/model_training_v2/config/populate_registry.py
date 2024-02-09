# ruff: noqa


from ..trainer.preprocessing.steps.test_row_filter_split import mock_regional_move_df


def populate_baseline_registry() -> None:
    """
    Populate the registry with all the registered functions

    It is also possible to do this using hooks, but this is more explicit
    and easier to debug for people who are not familiar with python setup hooks.
    """
    # Data
    from ..trainer.data.dataloaders import ParquetVerticalConcatenator, ParquetLoader

    # Estimator steps
    from ..trainer.task.estimator_steps.logistic_regression import logistic_regression_step
    from ..trainer.task.estimator_steps.xgboost import xgboost_classifier_step
    from ..trainer.task.estimator_steps.lightgbm import lightgbm_classifier_step

    # Preprocessing
    from ..trainer.preprocessing.pipeline import BaselinePreprocessingPipeline

    ### Cells
    from ..trainer.preprocessing.steps.cell_transformers import BoolToInt

    ### Rows
    from ..trainer.preprocessing.steps.row_filter_split import RegionalFilter
    from ..trainer.preprocessing.steps.row_filter_split import FilterByEntityID
    from ..trainer.preprocessing.steps.row_filter_other import AgeFilter

    ### Columns
    from ..trainer.preprocessing.steps.column_filters import (
        FilterColumnsWithinSubset,
        RegexColumnBlacklist,
        TemporalColumnFilter,
    )
    from ..trainer.preprocessing.steps.column_validator import (
        ColumnExistsValidator,
        ColumnPrefixExpectation,
    )

    # Loggers
    from ..loggers.terminal_logger import TerminalLogger
    from ..loggers.disk_logger import DiskLogger
    from ..loggers.multi_logger import MultiLogger
    from ..loggers.mlflow_logger import MLFlowLogger

    # Suggesters
    from ..hyperparameter_suggester.hyperparameter_suggester import SuggesterSpace
    from ..trainer.task.estimator_steps.logistic_regression import LogisticRegressionSuggester

    # Tasks
    from ..trainer.task.pipeline_constructor import pipeline_constructor
    from ..trainer.task.binary_classification.binary_classification_task import (
        BinaryClassificationTask,
    )
    from ..trainer.task.binary_classification.binary_classification_pipeline import (
        BinaryClassificationPipeline,
    )
    from ..trainer.task.binary_classification.binary_metrics.binary_auroc import BinaryAUROC

    # Trainers
    from ..trainer.cross_validator_trainer import CrossValidatorTrainer
    from ..trainer.split_trainer import SplitTrainer

    # Test data
    from ..trainer.preprocessing.steps.test_row_filter_split import mock_split_id_sequence
    from ..trainer.data.test_dataloaders import MinimalTestData
    from ..hyperparameter_suggester.test_optuna_hyperparameter_search import MockLogisticRegression
