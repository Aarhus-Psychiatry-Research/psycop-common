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
    def __init__(self, column_names: PatientColumnNames) -> None:
        self._column_names = column_names

    def _event_dict_to_event_obj(
        self,
        patient: Patient,
        event_row: dict[str, Any],
    ) -> TemporalEvent:
        return TemporalEvent(
            patient=patient,
            timestamp=event_row[self._column_names.timestamp_col_name],
            source=event_row[self._column_names.source_col_name],
            name=event_row[self._column_names.name_col_name]
            if self._column_names.name_col_name is not None
            else None,
            value=event_row[self._column_names.value_col_name],
        )

    def _patient_df_to_patient_obj(self, patient_events: pl.DataFrame) -> Patient:
        temporal_event_dicts = patient_events.iter_rows(named=True)

        first_row = next(patient_events.iter_rows(named=True))
        cur_patient = Patient(
            patient_id=first_row[self._column_names.patient_id_col_name],
        )

        temporal_event_objs = [
            self._event_dict_to_event_obj(event_row=e, patient=cur_patient)
            for e in temporal_event_dicts
        ]
        cur_patient.add_temporal_events(temporal_event_objs)

        return cur_patient

    def unpack(
        self,
        source_event_dataframe: pl.DataFrame,
    ) -> list[Patient]:
        patient_dfs = source_event_dataframe.partition_by(
            by=self._column_names.patient_id_col_name,
            maintain_order=True,
        )
        patient_objs = [
            self._patient_df_to_patient_obj(
                patient_df,
            )
            for patient_df in patient_dfs
        ]
        return patient_objs

    def unpack_static_events(self):
        pass
