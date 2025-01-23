from typing import Literal

from sklearn.feature_selection import SelectPercentile, chi2, f_classif, mutual_info_classif

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.task.model_step import ModelStep


@BaselineRegistry.estimator_steps.register("select_percentile")
def select_percentile(
    score_function_name: Literal["f_classif", "chi2", "mutual_info_classif"], percentile: int
) -> ModelStep:
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
