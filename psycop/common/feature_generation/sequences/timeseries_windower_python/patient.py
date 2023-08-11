import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

from psycop.common.feature_generation.sequences.timeseries_windower_python.events.static_event import (
    StaticEvent,
)
from psycop.common.feature_generation.sequences.timeseries_windower_python.events.temporal_event import (
    TemporalEvent,
)
from psycop.common.feature_generation.sequences.timeseries_windower_python.prediction_sequence import (
    PredictionSequence,
)


@dataclass(frozen=True)
class Patient:
    """All task-agnostic data for a patient."""

    patient_id: str | int
    temporal_events: Sequence[TemporalEvent]
    # TODO: Should we check that temporal events are sorted by timestamp on init?

    static_events: Sequence[StaticEvent] | None

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

    def to_prediction_sequences(
        self,
        lookbehind: dt.timedelta,
        outcome_timestamp: dt.datetime,
        prediction_timestamps: Sequence[dt.datetime],
    ) -> list[PredictionSequence]:
        # Map each prediction time to a prediction sequence:
        prediction_sequences = []

        for prediction_timestamp in prediction_timestamps:
            # 1. Filter the predictor events to those that are relevant to the prediction time. (Keep all static, drop all temporal that are outside the lookbehind window.)
            filtered_events = self._filter_events(
                events=self.temporal_events,
                start=prediction_timestamp - lookbehind,
                end=prediction_timestamp,
            )

            # 2. Return prediction sequences
            prediction_sequences.append(
                PredictionSequence(
                    patient_id=self.patient_id,
                    prediction_timestamp=prediction_timestamp,
                    temporal_events=filtered_events,
                    outcome_timestamp=outcome_timestamp,
                    static_events=self.static_events,
                ),
            )

        return prediction_sequences
