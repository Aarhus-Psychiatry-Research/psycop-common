from typing import Protocol

from .polars_frame import PolarsFrame


class PreprocessingPipeline(Protocol):
    def apply(self, data: PolarsFrame) -> PolarsFrame:
        ...
