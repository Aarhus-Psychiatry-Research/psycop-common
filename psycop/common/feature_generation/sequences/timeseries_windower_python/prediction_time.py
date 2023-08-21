from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Sequence

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
    static_features: Sequence[StaticFeature]
    prediction_timestamp: dt.datetime
    outcome: dt.datetime | None  # TODO: Maybe this should be a bool for the given lookahead window? Where do we want this parsing to happen?
