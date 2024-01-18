from dataclasses import dataclass

import pandas as pd
import polars as pl

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.task.base_metric import PredProbaSeries
from psycop.common.model_training_v2.trainer.task.base_task import BaselineTask
from psycop.common.model_training_v2.trainer.task.binary_classification.binary_classification_pipeline import (
    BinaryClassificationPipeline,
)
from psycop.common.types.polarsframe import PolarsFrame


def polarsframe_to_series(polarsframe: PolarsFrame) -> pl.Series:
    if isinstance(polarsframe, pl.LazyFrame):
        polarsframe = polarsframe.collect()

    assert len(polarsframe.columns) == 1

    return polarsframe.to_series()


@BaselineRegistry.tasks.register("binary_classification")
@dataclass(frozen=True)
class BinaryClassificationTask(BaselineTask):
    pipe: BinaryClassificationPipeline

    def train(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        y_col_name: str,
    ) -> None:
        assert len(y.columns) == 1
        y_series = y[y_col_name]

        self.pipe.fit(x=x, y=y_series)

    def predict_proba(self, x: pd.DataFrame) -> PredProbaSeries:
        return self.pipe.predict_proba(x)
