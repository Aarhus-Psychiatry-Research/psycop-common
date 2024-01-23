import polars as pl
from polars import Boolean

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.preprocessing.step import PresplitStep


@BaselineRegistry.preprocessing.register("bool_to_int")
class BoolToInt(PresplitStep):
    def __init__(self):
        pass

    def apply(self, input_df: pl.LazyFrame) -> pl.LazyFrame:
        for col_name in input_df.columns:
            if input_df.schema[col_name] == Boolean:  # type: ignore
                input_df = input_df.with_columns(pl.col(col_name).cast(int))

        return input_df
