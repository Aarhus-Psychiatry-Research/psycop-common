import sklearn
import sklearn.preprocessing

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.task.model_step import ModelStep


@BaselineRegistry.estimator_steps.register("standard_scaler")
def standard_scaler_step() -> ModelStep:
    return ("scaler", sklearn.preprocessing.StandardScaler())
