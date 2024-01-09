from typing import Literal

from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer

from psycop.common.model_training_v2.trainer.task.model_step import ModelStep
from psycop.projects.scz_bp.model_training.scz_bp_registry import SczBpRegistry


@SczBpRegistry.estimator_steps.register("simple_imputation")
def simple_imputation_step(
    strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean",
) -> ModelStep:
    return (
        "simple_imputation",
        SimpleImputer(
            strategy=strategy,
        ),
    )


@SczBpRegistry.estimator_steps.register("miss_forest_imputation")
def miss_forest_imputation_step() -> ModelStep:
    """Naive implementation of missforest using sklearn's IterativeImputer"""

    return (
        "miss_forest_imputation",
        IterativeImputer(
            estimator=RandomForestRegressor(),
            random_state=0,
        ),
    )
