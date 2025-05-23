from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime as dt

    from psycop.common.data_structures.patient import PatientSlice


@dataclass(frozen=True)
class PredictionTime:
    """A cut sequence of events for a patient, ready to issue a prediction."""

    prediction_timestamp: dt.datetime
    patient_slice: PatientSlice
    outcome: bool

    @property
    def pred_time_uuid(self) -> str:
        prediction_time_string = self.prediction_timestamp.strftime("%Y-%m-%d-%H-%M-%S")
        return f"{self.patient_slice.patient.patient_id}-{prediction_time_string}"
