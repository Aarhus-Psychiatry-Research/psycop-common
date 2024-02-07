import re

import polars as pl
import polars.selectors as cs

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.preprocessing.step import PresplitStep


@BaselineRegistry.preprocessing.register("lookbehind_combination_col_filter")
class LookbehindCombinationColFilter(PresplitStep):
    def __init__(self, lookbehinds: set[int], pred_col_prefix: str):
        self.lookbehinds = lookbehinds
        self.pred_col_prefix = pred_col_prefix
        self.lookbehind_pattern = r"within_(\d+)_days"

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        pred_cols_with_lookbehind = [
            col
            for col in input_df.columns
            if col.startswith(self.pred_col_prefix) and "within" in col
        ]

        lookbehinds_in_dataset = {
            int(re.findall(pattern=self.lookbehind_pattern, string=col)[0])
            for col in pred_cols_with_lookbehind
        }

        # Check that all loobehinds in lookbehind_combination are used in the predictors
        if not self.lookbehinds.issubset(lookbehinds_in_dataset):
            self.logger.warn(
                f"One or more of the provided lookbehinds in lookbehind_combination is/are not used in any predictors in the dataset: {self.lookbehinds - lookbehinds_in_dataset}"
            )

            lookbehinds_to_keep = self.lookbehinds.intersection(lookbehinds_in_dataset)

            if not lookbehinds_to_keep:
                self.logger.fail("No predictors left after dropping lookbehinds.")
                raise ValueError(
                    "Endng training because no predictors left after dropping lookbehinds."
                )

            self.logger.warn(f"Training on {lookbehinds_to_keep}.")
        else:
            lookbehinds_to_keep = self.lookbehinds

        cols_to_drop = self._cols_with_lookbehind_not_in_lookbehinds(
            pred_cols_with_lookbehind, lookbehinds_to_keep
        )

        return input_df.drop(columns=cols_to_drop)

    def _cols_with_lookbehind_not_in_lookbehinds(
        self, pred_cols_with_lookbehind: list[str], lookbehinds_to_keep: set[int]
    ) -> list[str]:
        """Identify columns that have a lookbehind that is not in lookbehinds_to_keep."""
        return [
            c
            for c in pred_cols_with_lookbehind
            if all(str(l_beh) not in c for l_beh in lookbehinds_to_keep)
        ]


@BaselineRegistry.preprocessing.register("temporal_col_filter")
class TemporalColumnFilter(PresplitStep):
    """Drops all temporal columns"""

    def __init__(self):
        pass

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        temporal_columns = input_df.select(cs.temporal()).columns
        return input_df.drop(temporal_columns)


@BaselineRegistry.preprocessing.register("regex_column_blacklist")
class RegexColumnBlacklist(PresplitStep):
    def __init__(self, *args: str):
        self.regex_blacklist = args

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        for blacklist in self.regex_blacklist:
            input_df = input_df.select(pl.exclude(f"^{blacklist}$"))

        return input_df


@BaselineRegistry.preprocessing.register("filter_columns_within_subset")
class FilterColumnsWithinSubset(PresplitStep):
    """Creates a subset matching one regex rule, and then within that subset, drops columns that do not match another rule"""

    def __init__(self, subset_rule: str, keep_matching: str):
        self.subset_rule = subset_rule
        self.keep_matching = keep_matching

        for rule in (subset_rule, keep_matching):
            try:
                re.compile(rule)
            except re.error as e:
                raise ValueError(f"Invalid regex rule: {rule}") from e

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        all_columns = input_df.columns
        subset_columns = [column for column in all_columns if re.match(self.subset_rule, column)]
        columns_to_drop = [
            column for column in subset_columns if not re.match(self.keep_matching, column)
        ]

        return input_df.drop(columns_to_drop)
