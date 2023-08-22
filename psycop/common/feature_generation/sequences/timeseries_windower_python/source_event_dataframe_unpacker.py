from dataclasses import dataclass
from typing import Any, Sequence

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


PatientDict = dict[str | int, list[TemporalEvent | StaticFeature]]

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

    def _temporal_event_dict_to_event_obj(
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

    def _static_feature_dict_to_event_obj(
        self,
        patient: Patient | None,
        event_row: dict[str, Any],
    ) -> StaticFeature:
        return StaticFeature(
            patient=patient,
            source=event_row[self._column_names.source_col_name],
            value=event_row[self._column_names.value_col_name],
        )

    def _patient_df_to_patient_dict(
        self,
        patient_events: pl.DataFrame,
    ) -> PatientDict:
        
        event_dicts = patient_events.iter_rows(named=True)
        

        is_temporal_events = self._column_names.timestamp_col_name in patient_events.columns

        if is_temporal_events:
            event_objects = [
                self._temporal_event_dict_to_event_obj(event_row=e, patient=None)
                for e in event_dicts
            ]

        else:
            event_objects = [
                self._static_feature_dict_to_event_obj(event_row=e, patient=None)
                for e in event_dicts
            ]

        first_row = next(patient_events.iter_rows(named=True))
        patient_id: str = first_row[self._column_names.patient_id_col_name]
        patient_dict = {patient_id: event_objects}

        return patient_dict # type: ignore

    def _cohort_dict_to_patients(self, cohort_dict: PatientDict) -> list[Patient]:
        patient_cohort = list()

        for patient_id, patient_events in cohort_dict.items():
            patient = Patient(patient_id=patient_id)
            patient.add_events(patient_events)
            patient_cohort.append(patient)

        return patient_cohort


    def unpack(
        self,
        source_event_dataframes: Sequence[pl.DataFrame],
    ) -> list[Patient]:
        patient_dfs_collections = [df.partition_by(
            by=self._column_names.patient_id_col_name,
            maintain_order=True,
        ) for df in source_event_dataframes]
        
        patient_dicts = [
            self._patient_df_to_patient_dict(
                patient_df,
            )
            for collection in patient_dfs_collections for patient_df in collection
        ] 
        
        cohort_dict = dict()
        for patient_dict in patient_dicts:
            patient_id = list(patient_dict.keys())[0]
            if patient_id not in cohort_dict.keys():
                cohort_dict.update(patient_dict)
            else:
                cohort_dict[patient_id] += list(patient_dict.values())

        patient_cohort = self._cohort_dict_to_patients(cohort_dict = cohort_dict)
                
        return patient_cohort

