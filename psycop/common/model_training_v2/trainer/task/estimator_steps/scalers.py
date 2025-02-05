import optuna
import sklearn
import sklearn.preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

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

    def fit(self, input_array, y=None):  # type: ignore # noqa
        return self

    def transform(self, input_array, y=None):  # type: ignore # noqa
        return input_array


@BaselineRegistry.estimator_steps.register("noop_scaler")
def noop_scaler_step() -> ModelStep:
    return ("scaler", IdentityTransformer())


@BaselineRegistry.estimator_steps.register("standard_scaler")
def standard_scaler_step() -> ModelStep:
    return ("scaler", sklearn.preprocessing.StandardScaler())


IMPLEMENTED_STRATEGIES = ["standard", "noop"]

STRATEGY2STEP = {"standard": "standard_scaler", "noop": "noop_scaler"}


@BaselineRegistry.estimator_steps_suggesters.register("scaler_suggester")
class ScalerSuggester(Suggester):
    def __init__(self, strategies: CategoricalSpaceT):
        for strategy in strategies:
            if strategy not in IMPLEMENTED_STRATEGIES:
                raise ValueError(f"Scaler strategy {strategy} is not implemented")

        self.strategy = CategoricalSpace(choices=strategies)

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, str]:
        strategy = self.strategy.suggest(trial, "scaler_strategy")
        estimator_step_str = STRATEGY2STEP[strategy]

        return {"@estimator_steps": estimator_step_str}
