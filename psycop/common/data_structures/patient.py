from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from psycop.common.data_structures.prediction_time import PredictionTime
from psycop.common.data_structures.static_feature import StaticFeature
from psycop.common.data_structures.temporal_event import TemporalEvent

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Sequence

    from psycop.common.feature_generation.sequences.prediction_times_from_cohort import (
        PATIENT_ID,
    )


@dataclass(frozen=True)
class TimeInterval:
    start: dt.datetime
    end: dt.datetime


@dataclass
class Patient:
    """All task-agnostic data for a patient."""

    patient_id: PATIENT_ID
    date_of_birth: dt.datetime
    unsorted_temporal_events: list[TemporalEvent] = field(default_factory=list)
    unsorted_static_features: list[StaticFeature] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"""
    patient_id: {self.patient_id}
    date_of_birth: {self.date_of_birth}
    n temporal_events: {len(self.unsorted_temporal_events)}
    n static_features: {len(self.unsorted_static_features)}"""

    @staticmethod
    def _filter_events_within_time_interval(
        events: Sequence[TemporalEvent],
        start: dt.datetime,
        end: dt.datetime,
    ) -> Sequence[TemporalEvent]:
        # This could be much faster if we assume that the events are already sorted.
        # Then we could implement binary search, which is O(log n) instead of O(n).
        # However, this might be plenty fast. We can always optimize later.
        return [event for event in events if start <= event.timestamp < end]

    def add_events(self, events: Sequence[TemporalEvent | StaticFeature]):
        # add patient reference to each event
        self.unsorted_temporal_events += [
            event for event in events if isinstance(event, TemporalEvent)
        ]
        self.unsorted_static_features += [
            event for event in events if isinstance(event, StaticFeature)
        ]

    @property
    def temporal_events(self) -> Sequence[TemporalEvent]:
        self.unsorted_temporal_events.sort(key=lambda event: event.timestamp)
        return self.unsorted_temporal_events

    @property
    def static_features(self) -> Sequence[StaticFeature]:
        return self.unsorted_static_features

    def as_slice(self) -> PatientSlice:
        """Returns the patient's entire history of temporal events."""
        return self.slice(time_interval=None)

    def slice(  # noqa: A003
        self,
        time_interval: TimeInterval | None,
    ) -> PatientSlice:
        """Creates a patient slice, i.e. a subset of the patient's data within a specific time interval."""
        if time_interval is not None:
            filtered_events = self._filter_events_within_time_interval(
                events=self.temporal_events,
                start=time_interval.start,
                end=time_interval.end,
            )
        else:
            filtered_events = self.temporal_events

        return PatientSlice(
            patient=self,
            temporal_events=filtered_events,
        )

    def to_prediction_times(
        self,
        lookbehind: dt.timedelta,
        lookahead: dt.timedelta,
        outcome_timestamp: dt.datetime | None,
        prediction_timestamps: Sequence[dt.datetime],
    ) -> list[PredictionTime]:
        """Creates prediction times for a boolean outome. E.g. for the task of predicting whether a patient will be diagnosed with diabetes within the next year, this function will return a list of PredictionTime objects, each of which contains the patient's data for a specific prediction time (predictors, prediction timestamp and whether the outcome occurs within the lookahead)."""
        # Map each prediction time to a prediction sequence:
        prediction_sequences = []

        for prediction_timestamp in prediction_timestamps:
            # 1. Filter the predictor events to those that are relevant to the prediction time. (Keep all static, drop all temporal that are outside the lookbehind window.)
            time_interval = TimeInterval(
                start=prediction_timestamp - lookbehind,
                end=prediction_timestamp,
            )
            patient_slice = self.slice(
                time_interval=time_interval,
            )

            outcome_within_lookahead = (
                outcome_timestamp <= (prediction_timestamp + lookahead)
                if outcome_timestamp is not None
                else False
            )

            # 2. Return prediction sequences
            prediction_sequences.append(
                PredictionTime(
                    patient_slice=patient_slice,
                    prediction_timestamp=prediction_timestamp,
                    outcome=outcome_within_lookahead,
                ),
            )

        return prediction_sequences


@dataclass(frozen=True, slots=True)
class PatientSlice:
    patient: Patient
    temporal_events: Sequence[TemporalEvent]
