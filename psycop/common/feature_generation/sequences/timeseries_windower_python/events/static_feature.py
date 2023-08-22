from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from psycop.common.feature_generation.sequences.timeseries_windower_python.patient import (
        Patient,
    )


@dataclass
class StaticFeature:
    soure_type: str  # E.g. "date-of-birth"/"gender"
    patient: Patient | None
    value: float | str | bool | datetime

    def __eq__(self, __value: StaticFeature) -> bool:  # type: ignore
        """Patient attr is ignored when computing equality. Needed for tests, since we have circular references in the dataclass, and the default __eq__ doesn't work."""
        for attr in self.__dict__:
            if attr == "patient":
                continue
            if getattr(self, attr) != getattr(__value, attr):
                return False
        return True
