from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Protocol

import polars as pl

from psycop.common.global_utils.pydantic_basemodel import PSYCOPBaseModel
from psycop.common.types.polarsframe import PolarsFrameGeneric


class EagerFilter(Protocol):
    """Interface for filtering functions applied to prediction times"""

    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        ...


class LazyFilter(Protocol):
    @staticmethod
    def apply(df: PolarsFrameGeneric) -> PolarsFrameGeneric:
        ...


class StepDelta(PSYCOPBaseModel):
    step_name: str
    n_prediction_times_before: int
    n_prediction_times_after: int
    n_ids_before: int
    n_ids_after: int
    step_index: int

    @property
    def n_dropped_prediction_times(self) -> int:
        return self.n_prediction_times_before - self.n_prediction_times_after

    @property
    def n_dropped_ids(self) -> int:
        return self.n_ids_before - self.n_ids_after


class FilteredPredictionTimeBundle(PSYCOPBaseModel):
    prediction_times: pl.DataFrame
    filter_steps: list[StepDelta]


class CohortDefiner(ABC):
    @staticmethod
    @abstractmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle:
        ...

    @staticmethod
    @abstractmethod
    def get_outcome_timestamps() -> pl.DataFrame:
        ...


def filter_prediction_times(
    prediction_times: pl.DataFrame,
    filtering_steps: Iterable[EagerFilter],
    entity_id_col_name: str,
) -> FilteredPredictionTimeBundle:
    stepdeltas: list[StepDelta] = []
    for i, filter_step in enumerate(filtering_steps):
        n_prediction_times_before = prediction_times.shape[0]
        n_ids_before = prediction_times[entity_id_col_name].n_unique()
        prediction_times = filter_step.apply(prediction_times)
        print(f"Applying filter: {filter_step.__class__.__name__}")
        stepdeltas.append(
            StepDelta(
                step_name=filter_step.__class__.__name__,
                n_prediction_times_before=n_prediction_times_before,
                n_prediction_times_after=prediction_times.shape[0],
                n_ids_before=n_ids_before,
                n_ids_after=prediction_times[entity_id_col_name].n_unique(),
                step_index=i,
            ),
        )

    if "date_of_birth" in prediction_times.columns:
        prediction_times = prediction_times.drop("date_of_birth")

    return FilteredPredictionTimeBundle(
        prediction_times=prediction_times,
        filter_steps=stepdeltas,
    )
