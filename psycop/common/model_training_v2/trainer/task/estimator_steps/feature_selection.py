from typing import Literal

import optuna

from sklearn.feature_selection import SelectPercentile, chi2, f_classif, mutual_info_classif

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.base_suggester import (
    Suggester,
)
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.suggester_spaces import (
    CategoricalSpace,
    CategoricalSpaceT,
)
from psycop.common.model_training_v2.trainer.task.model_step import ModelStep
from psycop.common.model_training_v2.trainer.task.estimator_steps.utils import IdentityTransformer


IMPLEMENTED_FUNCTIONS = Literal["f_classif", "chi2", "mutual_info_classif", "noop"]


@BaselineRegistry.estimator_steps.register("select_percentile")
def select_percentile(score_function_name: IMPLEMENTED_FUNCTIONS, percentile: int) -> ModelStep:
    match score_function_name:
        case "f_classif":
            return ("feature_selection", SelectPercentile(f_classif, percentile=percentile))
        case "chi2":
            return ("feature_selection", SelectPercentile(chi2, percentile=percentile))
        case "mutual_info_classif":
            return (
                "feature_selection",
                SelectPercentile(mutual_info_classif, percentile=percentile),
            )
        case "noop":
            return ("feature_selection", IdentityTransformer())


@BaselineRegistry.estimator_steps_suggesters.register("feature_selection_suggester")
class FeatureSelectionSuggester(Suggester):
    def __init__(self, score_functions: CategoricalSpaceT, percentiles: CategoricalSpaceT):
        self.score_function = CategoricalSpace(choices=score_functions)

        self.percentile = CategoricalSpace(choices=percentiles)

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, str]:
        score_function = self.score_function.suggest(trial, "score_function")
        percentile = self.percentile.suggest(trial, "percentile")

        return {
            "@estimator_steps": "select_percentile",
            "score_function_name": score_function,
            "percentile": percentile,
        }
