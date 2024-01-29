from abc import abstractmethod
from collections.abc import Sequence

from psycop.common.model_training_v2.trainer.task.model_step import ModelStep
from psycop.common.types.polarsframe import PolarsFrame

from ..base_pipeline import BasePipeline


class MultilabelClassificationPipeline(BasePipeline):
    @abstractmethod
    def __init__(self, steps: Sequence[ModelStep]) -> None:
        ...

    @abstractmethod
    def fit(self, X: PolarsFrame, y: PolarsFrame) -> None:
        ...

    @abstractmethod
    def predict_proba(self, X: PolarsFrame) -> PolarsFrame:
        ...
