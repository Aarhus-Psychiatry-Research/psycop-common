import datetime as dt
from collections import defaultdict

import polars as pl

from psycop.common.cohort_definition import CohortDefiner
from psycop.common.data_structures.patient import Patient
from psycop.common.data_structures.prediction_time import PredictionTime


class CohortToPredictionTimes:
    def __init__(self, cohort_definer: CohortDefiner, patient_objects: list[Patient]):
        self.cohort_definer = cohort_definer
        self.patients = patient_objects

    @staticmethod
    def _polars_dataframe_to_patient_timestamp_mapping(
        dataframe: pl.DataFrame,
        id_col_name: str,
        patient_timestamp_col_name: str,
    ) -> dict[str | int, list[dt.datetime]]:
        timestamp_dicts = dataframe.iter_rows(named=True)

        patient_to_prediction_times = defaultdict(list)
        for prediction_time_dict in timestamp_dicts:
            patient_id = prediction_time_dict[id_col_name]
            patient_to_prediction_times[patient_id].append(
                prediction_time_dict[patient_timestamp_col_name],
            )

        return patient_to_prediction_times

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

        prediction_times: list[PredictionTime] = []
        for patient in self.patients:
            pt_outcome_timestamps = outcome_timestamps.get(patient.patient_id)

            if pt_outcome_timestamps is not None:
                outcome_timestamp = pt_outcome_timestamps[0]
            else:
                outcome_timestamp = None

            prediction_times += patient.to_prediction_times(
                lookbehind=lookbehind,
                lookahead=lookahead,
                outcome_timestamp=outcome_timestamp,
                prediction_timestamps=prediction_timestamps[patient.patient_id],
            )

        return tuple(prediction_times)
