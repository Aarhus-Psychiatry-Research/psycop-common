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
