import pandas as pd
import polars as pl

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.trainer.preprocessing.polars_frame import (
    PolarsFrame,
)
from psycop.common.model_training_v2.trainer.task.base_task import BaselineTask
from psycop.common.model_training_v2.trainer.task.binary_classification.binary_classification_pipeline import (
    BinaryClassificationPipeline,
    PredProbaSeries,
)


def polarsframe_to_series(polarsframe: PolarsFrame) -> pl.Series:
    if isinstance(polarsframe, pl.LazyFrame):
        polarsframe = polarsframe.collect()

    assert len(polarsframe.columns) == 1

    return polarsframe.to_series()


@BaselineRegistry.tasks.register("binary_classification")
class BinaryClassification(BaselineTask):
    def __init__(
        self,
        task_pipe: BinaryClassificationPipeline,
        pred_time_uuid_col_name: str,
    ):
        self.pipe = task_pipe
        self.pred_time_uuid_col_name = pred_time_uuid_col_name

    def train(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        y_col_name: str,
    ) -> None:
        assert len(y.columns) == 1
        y_series = y[y_col_name]

        self.pipe.fit(x=x.drop(self.pred_time_uuid_col_name, axis=1), y=y_series)
        self.is_fitted = True

    def predict_proba(self, x: pd.DataFrame) -> PredProbaSeries:
        return self.pipe.predict_proba(x.drop(self.pred_time_uuid_col_name, axis=1))
