from typing import Literal

import optuna
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.base_suggester import (
    Suggester,
)
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.suggester_spaces import (
    CategoricalSpace,
    CategoricalSpaceT,
)
from psycop.common.model_training_v2.trainer.task.model_step import ModelStep


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):  # type: ignore
        return self

    def transform(self, input_array, y=None):  # type: ignore
        return input_array


@BaselineRegistry.estimator_steps.register("noop_imputer")
def noop_imputation_step() -> ModelStep:
    return ("imputer", IdentityTransformer())


@BaselineRegistry.estimator_steps.register("simple_imputation")
def simple_imputation_step(
    strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean",
) -> ModelStep:
    return ("imputer", SimpleImputer(strategy=strategy))


@BaselineRegistry.estimator_steps.register("miss_forest_imputation")
def miss_forest_imputation_step() -> ModelStep:
    """Naive implementation of missforest using sklearn's IterativeImputer"""

    return ("imputer", IterativeImputer(estimator=RandomForestRegressor(), random_state=0))


IMPLEMENTED_STRATEGIES = ["mean", "median", "most_frequent", "miss_forest", "noop"]

STRATEGY2STEP = {
    "mean": "simple_imputation",
    "median": "simple_imputation",
    "most_frequent": "simple_imputation",
    "miss_forest": "miss_forest_imputation",
    "noop": "noop",
}


@BaselineRegistry.estimator_steps_suggesters.register("imputation_suggester")
class ImputationSuggester(Suggester):
    def __init__(self, strategies: CategoricalSpaceT):
        for strategy in strategies:
            if strategy not in IMPLEMENTED_STRATEGIES:
                raise ValueError(f"Imputation strategy {strategy} is not implemented")

        self.strategy = CategoricalSpace(choices=strategies)

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, str]:
        strategy = self.strategy.suggest(trial, "imputation_strategy")
        estimator_step_str = STRATEGY2STEP[strategy]

        if strategy in ["miss_forest", "noop"]:
            return {"@estimator_steps": estimator_step_str}
        return {"@estimator_steps": estimator_step_str, "strategy": strategy}
