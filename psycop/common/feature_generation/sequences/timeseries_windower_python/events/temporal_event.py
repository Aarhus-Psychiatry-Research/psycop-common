import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalEvent:
    patient_id: int
    timestamp: dt.datetime
    # TODO: Add pointer to patient
    source: str  # E.g. "lab"/"diagnosis"
    name: str | None  # E.g. "Hba1c"/"hypertension"
    value: float | str  # 1/0 for booleans, numeric value for numeric events
