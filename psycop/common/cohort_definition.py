from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import polars as pl
from wasabi import Printer

from psycop.common.global_utils.pydantic_basemodel import PSYCOPBaseModel
from psycop.common.types.validated_frame import ValidatedFrame
from psycop.common.types.validator_rules import ColumnTypeRule, ValidatorRule


@dataclass(frozen=True)
class PredictionTimeFrame(ValidatedFrame[pl.DataFrame]):
    """ValidatedFrame with extra validation for prediction times"""

    frame: pl.DataFrame

    entity_id_col_name: str = "dw_ek_borger"
    entity_id_col_rules: Sequence[ValidatorRule] = (ColumnTypeRule(expected_type=pl.Int64),)

    timestamp_col_name: str = "timestamp"
    timestamp_col_rules: Sequence[ValidatorRule] = (ColumnTypeRule(expected_type=pl.Datetime),)

    allow_extra_columns: bool = True


@dataclass(frozen=True)
class OutcomeTimestampFrame(ValidatedFrame[pl.DataFrame]):
    """ValidatedFrame with extra validation for prediction times"""

    frame: pl.DataFrame

    entity_id_col_name: str = "dw_ek_borger"
    entity_id_col_rules: Sequence[ValidatorRule] = (ColumnTypeRule(expected_type=pl.Int64),)

    timestamp_col_name: str = "timestamp"
    timestamp_col_rules: Sequence[ValidatorRule] = (ColumnTypeRule(expected_type=pl.Datetime),)

    allow_extra_columns: bool = True


@runtime_checkable
class PredictionTimeFilter(Protocol):
    """Interface for filtering functions applied to prediction times"""

    def apply(self, df: pl.LazyFrame) -> pl.LazyFrame: ...


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


@dataclass(frozen=True)
class FilteredPredictionTimeBundle:
    prediction_times: PredictionTimeFrame
    filter_steps: list[StepDelta]


class CohortDefiner(ABC):
    @staticmethod
    @abstractmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle: ...

    @staticmethod
    @abstractmethod
    def get_outcome_timestamps() -> OutcomeTimestampFrame: ...


msg = Printer(timestamp=True)


def filter_prediction_times(
    prediction_times: pl.LazyFrame,
    filtering_steps: Iterable[PredictionTimeFilter],
    entity_id_col_name: str,
    timestamp_col_name: str = "timestamp",
    get_counts: bool = True,
) -> FilteredPredictionTimeBundle:
    """Apply a series of filters to prediction times.

    If prediction_times is a DataFrame, then the filters will be applied eagerly and we can calculate the stepdeltas.
    If not, then we will apply the filters lazily and we won't be able to calculate the stepdeltas.
    """

    stepdeltas: list[StepDelta] = []
    for i, filter_step in enumerate(filtering_steps):
        msg.info(f"Applying filter: {filter_step.__class__.__name__}")

        if get_counts:
            prefilter_prediction_times = prediction_times.collect()

        prediction_times = filter_step.apply(prediction_times)

        if get_counts:
            stepdeltas.append(
                StepDelta(
                    step_name=filter_step.__class__.__name__,
                    n_prediction_times_before=prefilter_prediction_times.shape[0],  # type: ignore
                    n_prediction_times_after=prediction_times.collect().shape[0],
                    n_ids_before=prefilter_prediction_times[entity_id_col_name].n_unique(),  # type: ignore
                    n_ids_after=prediction_times.collect()[entity_id_col_name].n_unique(),
                    step_index=i,
                )
            )
        # For much faster speeds, we can apply the filter step to the lazy frame.
        # This means we won't get the counts before and after. If you want those, be sure to pass in a DataFrame.

    if "date_of_birth" in prediction_times.columns:
        prediction_times = prediction_times.drop("date_of_birth")

    return FilteredPredictionTimeBundle(
        prediction_times=PredictionTimeFrame(frame=prediction_times.collect(), entity_id_col_name=entity_id_col_name, timestamp_col_name=timestamp_col_name),
        filter_steps=stepdeltas,
    )
