from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime as dt

    from psycop.common.feature_generation.sequences.timeseries_windower_python.patient import (
        Patient,
    )


@dataclass
class TemporalEvent:
    timestamp: dt.datetime
    source_type: str  # E.g. "lab"/"diagnosis"
    source_subtype: str | None  # E.g. "Hba1c"/"hypertension". Is optional, since e.g. diagnoses don't have a name, only a source and value.
    value: float | str | bool
    patient: Patient | None = None

    def __eq__(self, __value: TemporalEvent) -> bool:  # type: ignore
        """Patient attr is ignored when computing equality. Needed for tests, since we have circular references in the dataclass, and the default __eq__ doesn't work."""
        for attr in self.__dict__:
            if attr == "patient":
                continue
            if getattr(self, attr) != getattr(__value, attr):
                return False
        return True
