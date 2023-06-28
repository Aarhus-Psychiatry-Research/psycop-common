from abc import ABC, abstractmethod
from collections.abc import Iterable

import polars as pl

from psycop.common.global_utils.pydantic_basemodel import PSYCOPBaseModel


class PredictionTimeFilter(ABC):
    """Interface for filtering functions applied to prediction times. Requried
    to have an `apply` method that takes a `pl.DataFrame` (the unfiltered
    prediction times) and returns a `pl.DataFrame` (the filtered prediction times).
    """

    @staticmethod
    @abstractmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        ...


class StepDelta(PSYCOPBaseModel):
    step_name: str
    n_before: int
    n_after: int
    step_index: int

    @property
    def n_dropped(self) -> int:
        return self.n_before - self.n_after


class FilteredPredictionTimes(PSYCOPBaseModel):
    prediction_times: pl.DataFrame
    filter_steps: list[StepDelta]


class CohortDefiner(ABC):
    @staticmethod
    @abstractmethod
    def get_eligible_prediction_times() -> FilteredPredictionTimes:
        ...

    @staticmethod
    @abstractmethod
    def get_outcome_timestamps() -> pl.DataFrame:
        ...


def filter_prediction_times(
    prediction_times: pl.DataFrame,
    filtering_steps: Iterable[PredictionTimeFilter],
) -> FilteredPredictionTimes:
    stepdeltas: list[StepDelta] = []
    for i, filter_step in enumerate(filtering_steps):
        n_before = prediction_times.shape[0]
        prediction_times = filter_step.apply(prediction_times)
        stepdeltas.append(
            StepDelta(
                step_name=filter_step.__class__.__name__,
                n_before=n_before,
                n_after=prediction_times.shape[0],
                step_index=i,
            ),
        )

    return FilteredPredictionTimes(
        prediction_times=prediction_times,
        filter_steps=stepdeltas,
    )
