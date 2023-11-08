from typing import Protocol, TypeVar

from psycop.common.model_training_v2.training_method.preprocessing.polars_frame import (
    PolarsFrame,
)

PolarsFrame_T0 = TypeVar("PolarsFrame_T0", bound=PolarsFrame)


class PresplitStep(Protocol):
    def apply(self, input_df: PolarsFrame_T0) -> PolarsFrame_T0:
        ...
