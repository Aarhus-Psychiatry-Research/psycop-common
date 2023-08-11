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

    def _unpack_events(self, event_row: pl.DataFrame) -> list[TemporalEvent]:
        event_dict = event_row.to_dict()

    def _unpack_patient_events(self, patient_events: pl.DataFrame) -> Patient:
        events = patient_events.apply(self._unpack_events)

    def unpack(
        self,
        source_event_dataframe: pl.DataFrame,
        patient_id_col_name: str = "patient",
        timestamp_col_name: str = "timestamp",
        source_col_name: str = "source",
        value_col_name: str = "value",
        name_col_name: str | None = None,
    ) -> list[Patient]:
        patient_events = source_event_dataframe.groupby(patient_id_col_name).apply(
            self._unpack_patient_events
        )
        # Group by patient ID
        # Convert to a native python object
        # Parse to Patient objects
        pass
        return None

    def unpack_static_events(self):
        pass
