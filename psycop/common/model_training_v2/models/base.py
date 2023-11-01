from typing import Protocol

from ..presplit_preprocessing.step import PolarsFrame


class BaselineModel(Protocol):
    def fit(self, X_train: PolarsFrame, y_train: PolarsFrame):
        ...

    def predict_proba(self, X_val: PolarsFrame) -> PolarsFrame:
        ...
