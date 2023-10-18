from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from psycop.common.data_structures.patient_slice import PatientSlice

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Sequence

    from psycop.common.data_structures.patient import (
        Patient,
    )
    from psycop.common.data_structures.static_feature import StaticFeature
    from psycop.common.data_structures.temporal_event import TemporalEvent


@dataclass(frozen=True)
class PredictionTime:
    """A cut sequence of events for a patient, ready to issue a prediction."""

    patient_slice: PatientSlice
    temporal_events: Sequence[TemporalEvent]
    static_features: Sequence[StaticFeature]
    prediction_timestamp: dt.datetime
    outcome: bool
