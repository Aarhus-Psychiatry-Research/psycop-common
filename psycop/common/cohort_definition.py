from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

import pandas as pd
import polars as pl
from pydantic import BaseModel

from psycop.common.global_utils.pydantic_basemodel import PSYCOPBaseModel


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


class PredictionTimeFilter(ABC):
    """Interface for filtering functions applied to prediction times. Requried
    to have an `apply` method that takes a `pl.DataFrame` (the unfiltered
    prediction times) and returns a `pl.DataFrame` (the filtered prediction times).
    """

    @staticmethod
    @abstractmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        ...
