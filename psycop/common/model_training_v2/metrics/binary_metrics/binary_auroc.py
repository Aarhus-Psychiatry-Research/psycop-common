import polars as pl
from sklearn.metrics import roc_auc_score

from psycop.common.model_training_v2.metrics.binary_metrics.base_binary_metric import (
    BinaryMetric,
)
from psycop.common.model_training_v2.presplit_preprocessing.polars_frame import (
    PolarsFrame,
)


class BinaryAUROC(BinaryMetric):
    def __call__(self, y_true: PolarsFrame, y_pred: pl.Series) -> float:
        # sklearn expects MatrixLike | ArrayLike which PolarsFrame and pl.Series fulfill but are not typed as
        return roc_auc_score(y_true=y_true, y_score=y_pred)  # type: ignore
