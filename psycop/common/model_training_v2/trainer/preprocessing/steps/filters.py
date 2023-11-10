import re

import polars as pl

from psycop.common.model_training_v2.loggers.base_logger import BaselineLogger
from psycop.common.model_training_v2.trainer.preprocessing.step import (
    PolarsFrame_T0,
    PresplitStep,
)


class AgeFilter(PresplitStep):
    def __init__(self, min_age: int, max_age: int, age_col_name: str):
        self.min_age = min_age
        self.max_age = max_age
        self.age = pl.col(age_col_name)

    def apply(self, input_df: PolarsFrame_T0) -> PolarsFrame_T0:
        return input_df.filter((self.age >= self.min_age) & (self.age <= self.max_age))


class LookbehindCombinationFilter(PresplitStep):
    def __init__(
        self,
        lookbehind_combination: set[int],
        pred_col_prefix: str,
        logger: BaselineLogger,
    ):
        self.lookbehind_combination = lookbehind_combination
        self.pred_col_prefix = pred_col_prefix
        self.logger = logger

    def apply(self, input_df: PolarsFrame_T0) -> PolarsFrame_T0:
        pred_cols_with_lookbehind = [
            col
            for col in input_df.columns
            if col.startswith("pred_") and "within" in col
        ]

        lookbehinds_in_dataset = {
            int(re.findall(pattern=r"within_(\d+)_days", string=col)[0])
            for col in pred_cols_with_lookbehind
        }

        # Check that all loobehinds in lookbehind_combination are used in the predictors
        if not self.lookbehind_combination.issubset(
            lookbehinds_in_dataset,
        ):
            self.logger.warn(
                f"One or more of the provided lookbehinds in lookbehind_combination is/are not used in any predictors in the dataset: {self.lookbehind_combination - lookbehinds_in_dataset}",
            )

            lookbehinds_to_keep = self.lookbehind_combination.intersection(
                lookbehinds_in_dataset,
            )

            if not lookbehinds_to_keep:
                self.logger.fail("No predictors left after dropping lookbehinds.")

            self.logger.warn(f"Training on {lookbehinds_to_keep}.")
        else:
            lookbehinds_to_keep = self.lookbehind_combination

        cols_to_drop = [
            c
            for c in pred_cols_with_lookbehind
            if all(str(l_beh) not in c for l_beh in lookbehinds_to_keep)
        ]

        return input_df.drop(columns=cols_to_drop)
