import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import List

from psycop.common.feature_generation.sequences.timeseries_windower_python.events.static_feature import (
    StaticFeature,
)
from psycop.common.feature_generation.sequences.timeseries_windower_python.events.temporal_event import (
    TemporalEvent,
)
from psycop.common.feature_generation.sequences.timeseries_windower_python.prediction_time import (
    PredictionTime,
)


@dataclass
class Patient:
    """All task-agnostic data for a patient."""

    patient_id: str | int
    _temporal_events: List[TemporalEvent] = field(default_factory=list)
    _static_features: List[StaticFeature] = field(default_factory=list)

    @staticmethod
    def _filter_events(
        events: Sequence[TemporalEvent],
        start: dt.datetime,
        end: dt.datetime,
    ) -> Sequence[TemporalEvent]:
        # This could be much faster if we assume that the events are already sorted.
        # Then we could implement binary search, which is O(log n) instead of O(n).
        # However, this might be plenty fast. We can always optimize later.
        return [event for event in events if start <= event.timestamp < end]

    def add_temporal_events(self, events: List[TemporalEvent]):
        self._temporal_events += events

    def get_temporal_events(self) -> Sequence[TemporalEvent]:
        self._temporal_events.sort(key=lambda event: event.timestamp)
        return self._temporal_events

    def add_static_events(self, features: List[StaticFeature]):
        self._static_features += features

    def get_static_events(self) -> Sequence[StaticFeature]:
        return self._static_features

    def to_prediction_sequences(
        self,
        lookbehind: dt.timedelta,
        outcome_timestamp: dt.datetime,
        prediction_timestamps: Sequence[dt.datetime],
    ) -> list[PredictionTime]:
        # Map each prediction time to a prediction sequence:
        prediction_sequences = []

        for prediction_timestamp in prediction_timestamps:
            # 1. Filter the predictor events to those that are relevant to the prediction time. (Keep all static, drop all temporal that are outside the lookbehind window.)
            filtered_events = self._filter_events(
                events=self.get_temporal_events(),
                start=prediction_timestamp - lookbehind,
                end=prediction_timestamp,
            )

            # 2. Return prediction sequences
            prediction_sequences.append(
                PredictionTime(
                    patient=self,
                    prediction_timestamp=prediction_timestamp,
                    temporal_events=filtered_events,
                    outcome=outcome_timestamp,
                    static_features=self._static_features,
                ),
            )

        return prediction_sequences
