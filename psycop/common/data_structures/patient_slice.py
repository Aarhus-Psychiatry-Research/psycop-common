import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from psycop.common.data_structures.static_feature import StaticFeature
from psycop.common.data_structures.temporal_event import TemporalEvent

if TYPE_CHECKING:
    from psycop.common.data_structures.patient import Patient


@dataclass(frozen=True)
class PatientSlice:
    patient: Patient
    date_of_birth: dt.datetime
    _temporal_events: Sequence[TemporalEvent]
    _static_features: Sequence[StaticFeature]

    @property
    def temporal_events(self) -> Sequence[TemporalEvent]:
        temporal_events = list(self._temporal_events)
        temporal_events.sort(key=lambda event: event.timestamp)
        return temporal_events

    @property
    def static_features(self) -> Sequence[StaticFeature]:
        return self._static_features
