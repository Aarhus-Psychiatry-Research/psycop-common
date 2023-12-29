from typing import Literal

from sklearn.impute import SimpleImputer

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.task.model_step import ModelStep


@BaselineRegistry.estimator_steps.register("simple_imputation")
def simple_imputation_step(
    strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean",
) -> ModelStep:
    return (
        "simple_imputation",
        SimpleImputer(
            strategy=strategy,
        ),
    )
