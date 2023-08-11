from dataclasses import dataclass


@dataclass(frozen=True)
class StaticEvent:
    source: str  # E.g. "lab"/"diagnosis"
    name: str  # E.g. "Hba1c"/"hypertension"
    value: float | str  # 1/0 for booleans, numeric value for numeric events
