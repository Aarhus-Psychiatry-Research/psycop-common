from typing import Literal

from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer

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


@BaselineRegistry.estimator_steps.register("miss_forest_imputation")
def miss_forest_imputation_step() -> ModelStep:
    """Naive implementation of missforest using sklearn's IterativeImputer"""

    return (
        "miss_forest_imputation",
        IterativeImputer(
            estimator=RandomForestRegressor(),
            random_state=0,
        ),
    )
