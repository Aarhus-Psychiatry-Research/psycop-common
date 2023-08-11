import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

from psycop.common.feature_generation.sequences.timeseries_windower_python.events.static_feature import (
    StaticFeature,
)
from psycop.common.feature_generation.sequences.timeseries_windower_python.events.temporal_event import (
    TemporalEvent,
)
from psycop.common.feature_generation.sequences.timeseries_windower_python.patient import (
    Patient,
)


@dataclass(frozen=True)
class PredictionTime:
    """A cut sequence of events for a patient, ready to issue a prediction."""

    # TODO: rename to PredictionTime and add pointer to Patient and drop slices
    patient: Patient
    temporal_events: Sequence[TemporalEvent]
    prediction_timestamp: dt.datetime
    outcome: dt.datetime | None  # TODO: Maybe this should be a bool for the given lookahead window? Where do we want this parsing to happen?

    @property
    def static_features(self) -> Sequence[StaticFeature] | None:
        return self.patient.static_events
