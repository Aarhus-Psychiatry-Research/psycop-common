from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycop.common.feature_generation.sequences.timeseries_windower_python.patient import (
        Patient,
    )


@dataclass
class StaticFeature:
    source: str  # E.g. "date-of-birth"/"gender"
    patient: Patient | None
    value: float | str | bool
