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


@BaselineRegistry.suggesters.register("lookbehind_combination_filter_suggester")
class LookbehindCombinationFilterSuggester:
    def __init__(self, lookbehinds: CategoricalSpaceT, pred_col_prefix: str):
        self.lookbehinds = CategoricalSpace(lookbehinds)
        self.pred_col_prefix = pred_col_prefix

    def suggest_hyperparameters(self, trial: Trial) -> dict[str, str]:
        lookbehinds = self.lookbehinds.suggest(trial, "lookbehinds")
        return {
            "@preprocessing": "lookbehind_combination_col_filter",
            "lookbehinds": lookbehinds,
            "pred_col_prefix": self.pred_col_prefix,
        }


@BaselineRegistry.suggesters.register("blacklist_filter_suggester")
class BlacklistFilterSuggester:
    def __init__(self, regex_pattern: CategoricalSpaceT):
        self.regex_pattern = CategoricalSpace(choices=regex_pattern)

    def suggest_hyperparameters(self, trial: Trial) -> dict[str, str | list[str]]:
        regex_pattern = self.regex_pattern.suggest(trial, "regex_pattern")

        if regex_pattern == "noop":
            regex_pattern = "matchnothing"

        return {"@preprocessing": "regex_column_blacklist", "*": [regex_pattern]}
