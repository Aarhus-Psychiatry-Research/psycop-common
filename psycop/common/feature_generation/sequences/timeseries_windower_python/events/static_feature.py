from dataclasses import dataclass


@dataclass(frozen=True)
class StaticFeature:
    name: str  # E.g. "date-of-birth"/"gender"
    # TODO: Add pointer to patient
    value: float | str | bool  # 1/0 for booleans, numeric value for numeric events
