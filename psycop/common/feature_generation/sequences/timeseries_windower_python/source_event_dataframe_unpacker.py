from dataclasses import dataclass
from typing import Any

import polars as pl

from psycop.common.feature_generation.sequences.timeseries_windower_python.events.static_feature import (
    StaticFeature,
)
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
        patient: Patient | None,
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

    def _patient_df_to_patient_dict(
        self, patient_events: pl.DataFrame
    ) -> dict[str | int, list[TemporalEvent | StaticFeature]]:
        temporal_event_dicts = patient_events.iter_rows(named=True)

        first_row = next(patient_events.iter_rows(named=True))

        temporal_event_objs = [
            self._event_dict_to_event_obj(event_row=e, patient=None)
            for e in temporal_event_dicts
        ]

        patient_id: str = first_row[self._column_names.patient_id_col_name]
        patient_dict = {patient_id: temporal_event_objs}
        # TODO: we want to generalise this function to handle both temporal and static events
        return patient_dict  # type: ignore

    def unpack_temporal(
        self,
        source_event_dataframe: pl.DataFrame,
    ) -> list[Patient]:
        patient_dfs = source_event_dataframe.partition_by(
            by=self._column_names.patient_id_col_name,
            maintain_order=True,
        )
        patient_objs = [
            self._patient_df_to_patient_dict(
                patient_df,
            )
            for patient_df in patient_dfs
        ]
        return patient_objs

    def unpack_static(
        self,
        source_event_dataframe: pl.DataFrame,
        patients: list[Patient],
    ) -> list[Patient]:
        pass

    def unpack(
        self,
        static_source_dfs: list[pl.DataFrame],
        temporal_source_dfs: list[pl.DataFrame],
    ) -> list[Patient]:
        pass
