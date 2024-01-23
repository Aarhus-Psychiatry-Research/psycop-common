from sklearn.pipeline import Pipeline

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.task.model_step import ModelStep


@BaselineRegistry.task_pipelines.register("pipe_constructor")
def pipeline_constructor(*args: ModelStep) -> Pipeline:
    return Pipeline(args)
