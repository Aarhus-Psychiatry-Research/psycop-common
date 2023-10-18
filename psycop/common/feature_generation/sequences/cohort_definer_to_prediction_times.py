import datetime as dt
import itertools
from collections import defaultdict
from collections.abc import Sequence

import polars as pl

from psycop.common.cohort_definition import CohortDefiner
from psycop.common.data_structures.patient import PatientSlice
from psycop.common.data_structures.prediction_time import PredictionTime
from psycop.common.feature_generation.loaders.raw.load_ids import SplitName
from psycop.common.feature_generation.sequences.patient_loaders import (
    DiagnosisLoader,
    PatientLoader,
)
from psycop.projects.t2d.feature_generation.cohort_definition.t2d_cohort_definer import (
    T2DCohortDefiner,
)

PATIENT_ID = str | int


class CohortToPredictionTimes:
    def __init__(
        self, cohort_definer: CohortDefiner, patient_objects: list[PatientSlice]
    ):
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
        patient: PatientSlice,
        lookbehind: dt.timedelta,
        lookahead: dt.timedelta,
        outcome_timestamps: dict[PATIENT_ID, list[dt.datetime]],
        prediction_timestamps: dict[PATIENT_ID, list[dt.datetime]],
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


if __name__ == "__main__":
    patients = PatientLoader.get_split(
        event_loaders=[DiagnosisLoader(min_n_visits=5)],
        split=SplitName.TRAIN,
    )

    prediction_times = CohortToPredictionTimes(
        cohort_definer=T2DCohortDefiner(),
        patient_objects=patients,
    ).create_prediction_times(
        lookbehind=dt.timedelta(days=365),
        lookahead=dt.timedelta(days=365),
    )

    pass
