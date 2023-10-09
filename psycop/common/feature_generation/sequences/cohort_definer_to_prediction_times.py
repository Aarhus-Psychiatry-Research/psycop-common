import datetime as dt
import itertools
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import NewType

import polars as pl

from psycop.common.cohort_definition import CohortDefiner
from psycop.common.data_structures.patient import Patient
from psycop.common.data_structures.prediction_time import PredictionTime

PATIENT_ID = NewType("PATIENT_ID", str)


class CohortToPredictionTimes:
    def __init__(self, cohort_definer: CohortDefiner, patient_objects: list[Patient]):
        self.cohort_definer = cohort_definer
        self.patients = patient_objects

    @staticmethod
    def _polars_dataframe_to_patient_timestamp_mapping(
        dataframe: pl.DataFrame,
        id_col_name: str,
        patient_timestamp_col_name: str,
    ) -> dict[PATIENT_ID, list[dt.datetime]]:
        timestamp_dicts = dataframe.iter_rows(named=True)

        patient_to_prediction_times = defaultdict(list)
        for prediction_time_dict in timestamp_dicts:
            patient_id = prediction_time_dict[id_col_name]
            patient_to_prediction_times[patient_id].append(
                prediction_time_dict[patient_timestamp_col_name],
            )

        return patient_to_prediction_times

    @staticmethod
    def _patient_to_prediction_times(
        patient: Patient,
        lookbehind: dt.timedelta,
        lookahead: dt.timedelta,
        outcome_timestamps: Mapping[PATIENT_ID, list[dt.datetime]],
        prediction_timestamps: Mapping[PATIENT_ID, list[dt.datetime]],
    ) -> Sequence[PredictionTime]:
        pt_outcome_timestamps = outcome_timestamps.get(patient.patient_id)

        if pt_outcome_timestamps is not None:
            outcome_timestamp = pt_outcome_timestamps[0]
        else:
            outcome_timestamp = None

        return patient.to_prediction_times(
            lookbehind=lookbehind,
            lookahead=lookahead,
            outcome_timestamp=outcome_timestamp,
            prediction_timestamps=prediction_timestamps[patient.patient_id],
        )

    def create_prediction_times(
        self,
        lookbehind: dt.timedelta,
        lookahead: dt.timedelta,
    ) -> tuple[PredictionTime, ...]:
        outcome_timestamps = self._polars_dataframe_to_patient_timestamp_mapping(
            dataframe=self.cohort_definer.get_outcome_timestamps(),
            id_col_name="dw_ek_borger",
            patient_timestamp_col_name="timestamp",
        )
        prediction_timestamps = self._polars_dataframe_to_patient_timestamp_mapping(
            dataframe=self.cohort_definer.get_filtered_prediction_times_bundle().prediction_times,
            id_col_name="dw_ek_borger",
            patient_timestamp_col_name="timestamp",
        )

        prediction_times = (
            self._patient_to_prediction_times(
                patient=pt,
                lookbehind=lookbehind,
                lookahead=lookahead,
                outcome_timestamps=outcome_timestamps,
                prediction_timestamps=prediction_timestamps,
            )
            for pt in self.patients
        )

        return tuple(itertools.chain.from_iterable(prediction_times))
