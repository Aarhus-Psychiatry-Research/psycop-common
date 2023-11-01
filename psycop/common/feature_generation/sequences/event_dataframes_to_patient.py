from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl
from tqdm import tqdm
from wasabi import Printer

from psycop.common.data_structures.patient import Patient
from psycop.common.data_structures.static_feature import StaticFeature
from psycop.common.data_structures.temporal_event import TemporalEvent

msg = Printer(timestamp=True)

PatientDict = dict[str | int, list[TemporalEvent | StaticFeature]]


@dataclass(frozen=True)
class PatientSliceColumnNames:
    patient_id_col_name: str = "dw_ek_borger"
    timestamp_col_name: str = "timestamp"
    source_col_name: str = "source"
    value_col_name: str = "value"
    source_subtype_col_name: str | None = "type"


class EventDataFramesToPatientSlices:
    """Unpacks a sequence of dataframes containing events into a list of patients."""

    def __init__(self, column_names: PatientSliceColumnNames | None = None) -> None:
        self._column_names = (
            column_names if column_names is not None else PatientSliceColumnNames()
        )

    def _temporal_event_dict_to_event_obj(
        self,
        event_row: dict[str, Any],
    ) -> TemporalEvent:
        return TemporalEvent(
            timestamp=event_row[self._column_names.timestamp_col_name],
            source_type=event_row[self._column_names.source_col_name],
            source_subtype=event_row[self._column_names.source_subtype_col_name]
            if self._column_names.source_subtype_col_name is not None
            else None,
            value=event_row[self._column_names.value_col_name],
        )

    def _static_feature_dict_to_event_obj(
        self,
        event_row: dict[str, Any],
    ) -> StaticFeature:
        return StaticFeature(
            source_type=event_row[self._column_names.source_col_name],
            value=event_row[self._column_names.value_col_name],
        )

    def _patient_df_to_patient_dict(
        self,
        patient_events: pl.DataFrame,
    ) -> PatientDict:
        event_dicts = patient_events.iter_rows(named=True)

        is_temporal_events = (
            self._column_names.timestamp_col_name in patient_events.columns
        )

        if is_temporal_events:
            event_objects = [
                self._temporal_event_dict_to_event_obj(event_row=e) for e in event_dicts
            ]
        else:
            event_objects = [
                self._static_feature_dict_to_event_obj(event_row=e) for e in event_dicts
            ]

        patient_id: str = patient_events.select(
            pl.first(self._column_names.patient_id_col_name),
        ).item()

        patient_dict = {patient_id: event_objects}

        return patient_dict  # type: ignore

    def _cohort_dict_to_patient_slices(
        self,
        cohort_dict: PatientDict,
        date_of_birth_dict: dict[int | str, datetime],
    ) -> list[Patient]:
        patient_cohort: list[Patient] = []

        for patient_id, patient_events in cohort_dict.items():
            try:
                date_of_birth = date_of_birth_dict[patient_id]
            except KeyError as e:
                raise KeyError(
                    f"Patient {patient_id} does not have a date of birth. "
                    "Please make sure that the date of birth is included in the "
                    "date_of_birth_df.",
                ) from e
            patient = Patient(
                patient_id=patient_id,
                date_of_birth=date_of_birth,
            )
            patient.add_events(patient_events)
            patient_cohort.append(patient)

        return patient_cohort

    def _date_of_birth_df_to_dict(
        self,
        date_of_birth_df: pl.DataFrame,
    ) -> dict[int | str, datetime]:
        date_of_birth_dicts = date_of_birth_df.iter_rows(named=True)
        date_of_birth_dict = {
            row[self._column_names.patient_id_col_name]: row[
                self._column_names.timestamp_col_name
            ]
            for row in date_of_birth_dicts
        }

        return date_of_birth_dict

    def unpack(
        self,
        source_event_dataframes: Sequence[pl.DataFrame],
        date_of_birth_df: pl.DataFrame,
    ) -> list[Patient]:
        patient_dfs_collections = [
            df.partition_by(
                by=self._column_names.patient_id_col_name,
                maintain_order=True,
            )
            for df in source_event_dataframes
        ]

        patient_dicts = []
        for i, collection in enumerate(patient_dfs_collections):
            msg.info(f"Unpacking loader {i+1} of {len(patient_dfs_collections)}")

            for patient_df in tqdm(collection):
                patient_dicts.append(self._patient_df_to_patient_dict(patient_df))

        cohort_dict = {}

        for patient_dict in patient_dicts:
            patient_id = list(patient_dict.keys())[0]
            if patient_id not in cohort_dict.keys():
                cohort_dict.update(patient_dict)
            else:
                patient_events = list(patient_dict.values())[0]
                cohort_dict[patient_id] += patient_events

        date_of_birth_dict = self._date_of_birth_df_to_dict(
            date_of_birth_df=date_of_birth_df,
        )

        patient_cohort = self._cohort_dict_to_patient_slices(
            cohort_dict=cohort_dict,
            date_of_birth_dict=date_of_birth_dict,
        )

        return patient_cohort
