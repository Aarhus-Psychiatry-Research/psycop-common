import datetime as dt
from dataclasses import dataclass
from typing import Sequence

from psycop.common.feature_generation.sequences.timeseries_windower_python.static_event import (
    StaticEvent,
)
from psycop.common.feature_generation.sequences.timeseries_windower_python.temporal_event import (
    TemporalEvent,
)


@dataclass(frozen=True)
class PredictionSequence:
    """A cut sequence of events for a patient, ready to issue a prediction."""

    patient_id: str | int
    temporal_events: Sequence[TemporalEvent]
    static_events: Sequence[StaticEvent] | None
    prediction_timestamp: dt.datetime
    outcome_timestamp: dt.datetime | None  # TODO: Maybe this should be a bool for the given lookahead window? Where do we want this parsing to happen?
