from imblearn.pipeline import Pipeline
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry

from psycop.common.model_training_v2.trainer.task.model_step import ModelStep
from psycop.projects.scz_bp.model_training.scz_bp_registry import SczBpRegistry


@BaselineRegistry.task_pipelines.register("imblearn_pipe_constructor")
def imblearn_pipeline_constructor(*args: ModelStep) -> Pipeline:
    return Pipeline(args)
