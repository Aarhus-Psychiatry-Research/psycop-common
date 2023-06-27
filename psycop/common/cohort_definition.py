from abc import ABC, abstractmethod
from dataclasses import dataclass

import polars as pl
from pydantic import BaseModel


@dataclass(frozen=True)
class StepDelta:
    step_name: str
    n_before: int
    n_after: int
    step_index: int

    @property
    def n_dropped(self) -> int:
        return self.n_before - self.n_after


class FilteredPredictionTimes(BaseModel):
    """A filtered cohort is a cohort that has been filtered by a set of criteria."""

    prediction_times: pl.DataFrame
    filter_steps: list[StepDelta]

    class Config:
        arbitrary_types_allowed = True


class CohortDefiner(ABC):
    """Interface for a cohort definition."""

    @staticmethod
    @abstractmethod
    def get_eligible_prediction_times() -> FilteredPredictionTimes:
        ...

    @staticmethod
    @abstractmethod
    def get_outcome_timestamps() -> pl.DataFrame:
        ...


class PredictionTimeFilter(ABC):
    """Interface for filtering functions applied to prediction times. Requried
    to have an `apply` method that takes a `pl.DataFrame` (the unfiltered 
    prediction times) and returns a `pl.DataFrame` (the filtered prediction times).
    """

    @staticmethod
    @abstractmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        ...
