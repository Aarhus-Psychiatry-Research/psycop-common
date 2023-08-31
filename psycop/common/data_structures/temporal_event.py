from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime as dt


@dataclass
class TemporalEvent:
    timestamp: dt.datetime
    source_type: str  # E.g. "lab"/"diagnosis"
    source_subtype: str | None  # E.g. "Hba1c"/"hypertension". Is optional, since e.g. diagnoses don't have a name, only a source and value.
    value: float | str | bool
