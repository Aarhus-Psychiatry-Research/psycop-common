from dataclasses import dataclass
from typing import Any

import polars as pl

from psycop.common.feature_generation.sequences.timeseries_windower_python.events.temporal_event import (
    TemporalEvent,
)
from psycop.common.feature_generation.sequences.timeseries_windower_python.patient import (
    Patient,
)


@dataclass(frozen=True)
class PatientColumnNames:
    patient_id_col_name: str = "patient"
    timestamp_col_name: str = "timestamp"
    source_col_name: str = "source"
    value_col_name: str = "value"
    name_col_name: str | None = None


class SourceEventDataframeUnpacker:
    def __init__(self, column_names: PatientColumnNames | None) -> None:
        self._column_names = (
            PatientColumnNames() if column_names is None else column_names
        )

    def _unpack_events(self, event_row: dict[str, Any]) -> TemporalEvent:
        return TemporalEvent(
            patient=event_row[self._column_names.patient_id_col_name],
            timestamp=event_row[self._column_names.timestamp_col_name],
            source=event_row[self._column_names.source_col_name],
            name=None,  # TODO: Will this remain None?
            value=event_row[self._column_names.value_col_name],
        )

    def _unpack_patient_events(self, patient_events: pl.DataFrame) -> Patient:
        temporal_events = patient_events.iter_rows(named=True)
        unpacked_events = [self._unpack_events(e) for e in temporal_events]

        first_row = next(temporal_events)
        return Patient(
            patient_id=first_row[self._column_names.patient_id_col_name],
            _temporal_events=unpacked_events,
            _static_features=[],
            # TODO: Add static event unpacking
        )

    def unpack(
        self,
        source_event_dataframe: pl.DataFrame,
    ) -> list[Patient]:
        patient_dfs = source_event_dataframe.partition_by(
            by=self._column_names.patient_id_col_name, maintain_order=True
        )
        unpacked_patients = [
            self._unpack_patient_events(
                patient_df,
            )
            for patient_df in patient_dfs
        ]
        # Group by patient ID
        # Convert to a native python object
        # Parse to Patient objects
        return unpacked_patients

    def unpack_static_events(self):
        pass
