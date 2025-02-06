from optuna import Trial

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.suggester_spaces import (
    CategoricalSpace,
    CategoricalSpaceT,
)


@BaselineRegistry.suggesters.register("sufficient_window_filter_suggester")
class SufficientWindowFilterSuggester:
    def __init__(self, timestamp_col_name: str, n_days: CategoricalSpaceT, direction: str):
        self.timestamp_col_name = timestamp_col_name
        self.n_days = CategoricalSpace(choices=n_days)
        self.direction = direction

    def suggest_hyperparameters(self, trial: Trial) -> dict[str, str]:
        n_days = self.n_days.suggest(trial, "n_days")
        return {
            "@preprocessing": "window_filter",
            "timestamp_col_name": self.timestamp_col_name,
            "n_days": n_days,
            "direction": self.direction,
        }
