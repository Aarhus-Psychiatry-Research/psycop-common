from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime as dt

if TYPE_CHECKING:
    from psycop.common.feature_generation.sequences.timeseries_windower_python.patient import (
        Patient,
    )


@dataclass(frozen=True)
class TemporalEvent:
    patient: Patient
    timestamp: dt.datetime
    source: str  # E.g. "lab"/"diagnosis"
    name: str | None  # E.g. "Hba1c"/"hypertension"
    value: float | str  # 1/0 for booleans, numeric value for numeric events
