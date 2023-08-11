import polars as pl

from psycop.common.feature_generation.sequences.timeseries_windower_python.events.temporal_event import (
    TemporalEvent,
)
from psycop.common.feature_generation.sequences.timeseries_windower_python.patient import (
    Patient,
)


class SourceEventDataframeUnpacker:
    def __init__(self):
        pass

    def _unpack_events(self, event_row: pl.DataFrame) -> tuple[TemporalEvent]:
        return (TemporalEvent(
            timestamp=event_row[1],
            source=event_row[2],
            name=None,
            value=event_row[3],
        ),)


    def _unpack_patient_events(self, patient_events: pl.DataFrame) -> Patient:
        temporal_events=patient_events.apply(self._unpack_events)

        return Patient(
            patient_id=patient_events[0],
            temporal_events=temporal_events,
            static_events=None,
        )

    def unpack(
        self,
        source_event_dataframe: pl.DataFrame,
        patient_id_col_name: str = "patient",
        timestamp_col_name: str = "timestamp",
        source_col_name: str = "source",
        value_col_name: str = "value",
        name_col_name: str | None = None,
    ) -> list[Patient]:
        patients = source_event_dataframe.partition_by(patient_id_col_name, maintain_order=True, as_dict=True)
        unpacked_patients = [self._unpack_patient_events(patient) for patient in patients]
        # Group by patient ID
        # Convert to a native python object
        # Parse to Patient objects
        pass
        return None

    def unpack_static_events(self):
        pass
