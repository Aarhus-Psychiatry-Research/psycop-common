
def populate_baseline_registry() -> None:
    """
    Populate the registry with all the registered functions

    It is also possible to do this using hooks, but this is more explicit
    and easier to debug for people who are not familiar with python setup hooks.
    """
    from ..loggers.base_logger import TerminalLogger # noqa
    from psycop.common.model_training_v2.trainer.task.pipeline_constructor import pipeline_constructor # noqa


    