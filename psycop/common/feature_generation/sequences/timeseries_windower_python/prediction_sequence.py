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
    patient_id: str | int
    static_features: Sequence[StaticEvent]
    predictor_events: Sequence[TemporalEvent]
    prediction_timestamp: dt.datetime
    outcome_timestamp: dt.datetime | None
