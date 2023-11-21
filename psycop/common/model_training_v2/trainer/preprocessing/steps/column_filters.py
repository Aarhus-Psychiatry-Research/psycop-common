import re

import polars as pl

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.loggers.base_logger import (
    BaselineLogger,
)
from psycop.common.model_training_v2.trainer.preprocessing.step import (
    PolarsFrame_T0,
    PresplitStep,
)


@BaselineRegistry.preprocessing.register("lookbehind_combination_col_filter")
class LookbehindCombinationColFilter(PresplitStep):
    def __init__(
        self,
        lookbehinds: set[int],
        pred_col_prefix: str,
        logger: BaselineLogger,
    ):
        self.lookbehinds = lookbehinds
        self.pred_col_prefix = pred_col_prefix
        self.lookbehind_pattern = r"within_(\d+)_days"
        self.logger = logger

    def apply(self, input_df: PolarsFrame_T0) -> PolarsFrame_T0:
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
        if not self.lookbehinds.issubset(
            lookbehinds_in_dataset,
        ):
            self.logger.warn(
                f"One or more of the provided lookbehinds in lookbehind_combination is/are not used in any predictors in the dataset: {self.lookbehinds - lookbehinds_in_dataset}",
            )

            lookbehinds_to_keep = self.lookbehinds.intersection(
                lookbehinds_in_dataset,
            )

            if not lookbehinds_to_keep:
                self.logger.fail("No predictors left after dropping lookbehinds.")
                raise ValueError(
                    "Endng training because no predictors left after dropping lookbehinds.",
                )

            self.logger.warn(f"Training on {lookbehinds_to_keep}.")
        else:
            lookbehinds_to_keep = self.lookbehinds

        cols_to_drop = self._cols_with_lookbehind_not_in_lookbehinds(
            pred_cols_with_lookbehind,
            lookbehinds_to_keep,
        )

        return input_df.drop(columns=cols_to_drop)

    def _cols_with_lookbehind_not_in_lookbehinds(
        self,
        pred_cols_with_lookbehind: list[str],
        lookbehinds_to_keep: set[int],
    ) -> list[str]:
        """Identify columns that have a lookbehind that is not in lookbehinds_to_keep."""
        return [
            c
            for c in pred_cols_with_lookbehind
            if all(str(l_beh) not in c for l_beh in lookbehinds_to_keep)
        ]


@BaselineRegistry.preprocessing.register("regex_column_blacklist")
class RegexColumnBlacklist(PresplitStep):
    def __init__(self, *args: str):
        self.regex_blacklist = args

    def apply(self, input_df: PolarsFrame_T0) -> PolarsFrame_T0:
        for blacklist in self.regex_blacklist:
            input_df = input_df.select(pl.exclude(f"^{blacklist}$"))

        return input_df
