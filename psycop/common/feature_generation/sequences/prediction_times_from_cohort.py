import datetime as dt
import itertools
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from psycop.common.cohort_definition import CohortDefiner
from psycop.common.data_structures.patient import Patient
from psycop.common.data_structures.prediction_time import PredictionTime
from psycop.common.feature_generation.sequences.event_loader import DiagnosisLoader
from psycop.common.feature_generation.sequences.patient_loader import (
    PatientLoader,
)
from psycop.projects.t2d.feature_generation.cohort_definition.t2d_cohort_definer import (
    T2DCohortDefiner,
)

from ...model_training_v2.trainer.data.data_filters.geography import RegionalFilter
from ...model_training_v2.trainer.preprocessing.step import PresplitStep

PATIENT_ID = str | int


@dataclass(frozen=True)
class PredictionTimesFromCohort:
    cohort_definer: CohortDefiner
    patients: Sequence[Patient]
    split_filter: PresplitStep

    @staticmethod
    def _polars_dataframe_to_patient_timestamp_mapping(
        dataframe: pl.DataFrame,
        id_col_name: str,
        patient_timestamp_col_name: str,
        lookahead: dt.timedelta | None = None,
    ) -> dict[PATIENT_ID, list[dt.datetime]]:
        """Maps a polars dataframe to a dictionary of {patient ids: timestamps}.

        Args:
            dataframe: Polars dataframe with patient ids and timestamps.
            id_col_name: Name of the column with patient ids.
            patient_timestamp_col_name: Name of the column with timestamps.
            lookahead: If not None, only timestamps before the max timestamp minus the lookahead
                will be included. E.g. for a lookahead of 2 years, this ensures we actually have 2
                years of data to label the outcome.
        """
        if lookahead is not None:
            max_timestamp: dt.datetime = dataframe[patient_timestamp_col_name].max()  # type: ignore
            dataframe = dataframe.filter(
                pl.col(patient_timestamp_col_name) <= max_timestamp - lookahead,
            )

        timestamp_dicts = dataframe.iter_rows(named=True)

        patient_to_prediction_times = defaultdict(list)
        for prediction_time_dict in timestamp_dicts:
            patient_timestamp: dt.datetime = prediction_time_dict[
                patient_timestamp_col_name
            ]
            patient_id = prediction_time_dict[id_col_name]
            patient_to_prediction_times[patient_id].append(
                patient_timestamp,
            )

        return patient_to_prediction_times

    @staticmethod
    def _patient_to_prediction_times(
        patient: Patient,
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

        naive_prediction_times = (
            self.cohort_definer.get_filtered_prediction_times_bundle().prediction_times
        ).lazy()
        prediction_times_for_split = self.split_filter.apply(
            naive_prediction_times,
        ).collect()

        prediction_timestamps = self._polars_dataframe_to_patient_timestamp_mapping(
            dataframe=prediction_times_for_split,
            id_col_name="dw_ek_borger",
            patient_timestamp_col_name="timestamp",
            lookahead=lookahead,
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
    patients = PatientLoader(
        event_loaders=[DiagnosisLoader()],
        min_n_events=5,
    ).get_patients()

    prediction_times = PredictionTimesFromCohort(
        cohort_definer=T2DCohortDefiner(),
        patients=patients,
        split_filter=RegionalFilter(),
    ).create_prediction_times(
        lookbehind=dt.timedelta(days=365),
        lookahead=dt.timedelta(days=365),
    )
