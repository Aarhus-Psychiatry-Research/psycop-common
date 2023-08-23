import datetime as dt

from psycop.common.data_structures import Patient, TemporalEvent


class TestPatientSequenceGenerator:
    def test_temporal_event_filtering(self):
        """Temporal event filtering should remove predictor events which are before the start of the lookbehind window, and after the prediction time."""
        temporal_events = [
            TemporalEvent(
                timestamp=dt.datetime(2021, 1, 1),
                value=1,
                source_type="test_source",
                source_subtype="test_name",
            ),
            TemporalEvent(
                timestamp=dt.datetime(2021, 1, 3),
                value=2,
                source_type="test_source",
                source_subtype="test_name",
            ),
        ]

        patient = Patient(
            patient_id=1,
        )
        patient.add_events(temporal_events)  # type: ignore

        prediction_sequences = patient.to_prediction_times(
            lookbehind=dt.timedelta(days=2),
            lookahead=dt.timedelta(days=2),
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

        outcome_not_within_lookahead = prediction_sequences[0].outcome is False
        assert outcome_not_within_lookahead

        outcome_within_lookahead = prediction_sequences[1].outcome is True
        assert outcome_within_lookahead
