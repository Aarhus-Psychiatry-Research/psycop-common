from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycop.common.feature_generation.sequences.timeseries_windower_python.patient import (
        Patient,
    )


@dataclass(frozen=True)
class StaticFeature:
    name: str  # E.g. "date-of-birth"/"gender"
    patient: Patient
    value: float | str | bool
