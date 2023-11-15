def populate_baseline_registry() -> None:
    """
    Populate the registry with all the registered functions

    It is also possible to do this using hooks, but this is more explicit
    and easier to debug for people who are not familiar with python setup hooks.
    """
    from ..loggers.base_logger import TerminalLogger  # noqa
    from ..trainer.task.pipeline_constructor import pipeline_constructor  # noqa
    from ..trainer.task.estimator_steps.logistic_regression import (
        logistic_regression_step,  # noqa
    )
    from ..trainer.preprocessing.pipeline import BaselinePreprocessingPipeline  # noqa
    from ..trainer.preprocessing.steps.row_filters import AgeFilter  # noqa
    from ..trainer.cross_validator_trainer import CrossValidatorTrainer  # noqa
    from ..trainer.split_trainer import SplitTrainer  # noqa
    from ..trainer.task.binary_classification.binary_metrics.binary_auroc import (
        BinaryAUROC,  # noqa
    )
    from ..trainer.data.minimal_test_data import MinimalTestData  # noqa
    from ..trainer.task.binary_classification.binary_classification_pipeline import (
        BinaryClassificationPipeline,  # noqa
    )
    from ..trainer.task.binary_classification.binary_classification_task import (
        BinaryClassificationTask,  # noqa
    )
