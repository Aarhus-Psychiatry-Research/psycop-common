import polars as pl

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.preprocessing.step import (
    PolarsFrame_T0,
    PresplitStep,
)


@BaselineRegistry.preprocessing.register("regex_column_blacklist")
class RegexColumnBlacklist(PresplitStep):
    def __init__(self, *args: str):
        self.regex_blacklist = args

    def apply(self, input_df: PolarsFrame_T0) -> PolarsFrame_T0:
        for blacklist in self.regex_blacklist:
            input_df = input_df.select(pl.exclude(f"^{blacklist}$"))

        return input_df
