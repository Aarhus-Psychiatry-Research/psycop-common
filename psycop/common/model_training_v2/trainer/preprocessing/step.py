from typing import Protocol, TypeVar, runtime_checkable

from psycop.common.model_training_v2.trainer.preprocessing.polars_frame import (
    PolarsFrame,
)

PolarsFrame_T0 = TypeVar("PolarsFrame_T0", bound=PolarsFrame)


@runtime_checkable
class PresplitStep(Protocol):
    def apply(self, input_df: PolarsFrame_T0) -> PolarsFrame_T0:
        ...
