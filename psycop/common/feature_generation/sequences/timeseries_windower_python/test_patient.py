import datetime as dt

from psycop.common.feature_generation.sequences.timeseries_windower_python.events.temporal_event import (
    TemporalEvent,
)
from psycop.common.feature_generation.sequences.timeseries_windower_python.patient import (
    Patient,
)


class TestPatientSequenceGenerator:
    def test_temporal_event_filtering(self):
        """Temporal event filtering should remove predictor events which are before the start of the lookbehind window, and after the prediction time."""
        temporal_events = [
            TemporalEvent(
                timestamp=dt.datetime(2021, 1, 1),
                value=1,
                source="test_source",
                name="test_name",
            ),
            TemporalEvent(
                timestamp=dt.datetime(2021, 1, 3),
                value=2,
                source="test_source",
                name="test_name",
            ),
        ]

        patient = Patient(
            patient_id=1,
        )
        patient.add_temporal_events(temporal_events)

        prediction_sequences = patient.to_prediction_times(
            lookbehind=dt.timedelta(days=2),
            prediction_timestamps=[dt.datetime(2021, 1, 2), dt.datetime(2021, 1, 4)],
            outcome_timestamp=dt.datetime(2021, 1, 5),
        )

        exclude_events_after_prediction_time = prediction_sequences[
            0
        ].temporal_events == [temporal_events[0]]
        assert exclude_events_after_prediction_time

        exclude_events_before_start_of_lookbehind = prediction_sequences[
            1
        ].temporal_events == [
            temporal_events[1],
        ]
        assert exclude_events_before_start_of_lookbehind
