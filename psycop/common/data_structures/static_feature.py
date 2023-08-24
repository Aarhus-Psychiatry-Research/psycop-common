from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from psycop.common.data_structures.patient import (
        Patient,
    )


@dataclass
class StaticFeature:
    source_type: str  # E.g. "date-of-birth"/"gender"
    value: float | str | bool | datetime
