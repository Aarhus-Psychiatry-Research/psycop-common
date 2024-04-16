import pandas as pd
import polars as pl
from tqdm import tqdm

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    PredictionTimeFrame,
    filter_prediction_times,
)
from psycop.common.feature_generation.loaders.raw.load_visits import (
    get_time_of_first_visit_to_psychiatry,
)
from psycop.projects.bipolar.cohort_definition.diagnosis_specification.first_bipolar_diagnosis import (
    get_first_bipolar_diagnosis,
)
from psycop.projects.bipolar.cohort_definition.eligible_data.single_filters import (
    BipolarMinAgeFilter,
    BipolarMinDateFilter,
    BipolarPatientsWithF20F25Filter,
    BipolarWashoutMove,
)


def generate_timestamps(
    first_visit_date: pd.Timestamp, diagnosis_date: pd.Timestamp, interval_days: int = 30
) -> list:  # type: ignore
    timestamps = [diagnosis_date]
    current_date = diagnosis_date
    while current_date > (first_visit_date + pd.to_timedelta(interval_days, "d")):
        current_date -= pd.Timedelta(days=interval_days)
        timestamps.append(current_date)
    return timestamps[::-1]


class BipolarCohortDefiner(CohortDefiner):
    @staticmethod
    def get_bipolar_cohort(interval_days: int = 30) -> FilteredPredictionTimeBundle:
        bipolar_diagnosis_timestamps = pl.from_pandas(get_first_bipolar_diagnosis())

        filtered_bipolar_diagnosis_timestamps = filter_prediction_times(
            prediction_times=bipolar_diagnosis_timestamps.lazy(),
            filtering_steps=(
                BipolarMinDateFilter(),
                BipolarMinAgeFilter(),
                BipolarWashoutMove(),
                BipolarPatientsWithF20F25Filter(),
            ),
            entity_id_col_name="dw_ek_borger",
        )

        filtered_bipolar_diagnosis_timestamps_df = pd.DataFrame(
            filtered_bipolar_diagnosis_timestamps.prediction_times.frame.to_pandas()
        )

        first_visits_to_psychiatry = pd.DataFrame(
            get_time_of_first_visit_to_psychiatry().to_pandas()
        )

        filtered_bipolar_diagnosis_timestamps_df = filtered_bipolar_diagnosis_timestamps_df.merge(
            first_visits_to_psychiatry,
            on="dw_ek_borger",
            how="left",
            suffixes=(None, "_first_visit"),
        )

        filtered_bipolar_diagnosis_timestamps_df = filtered_bipolar_diagnosis_timestamps_df.dropna(
            subset=["timestamp_first_visit"]
        )

        filtered_bipolar_diagnosis_timestamps_df = filtered_bipolar_diagnosis_timestamps_df[
            filtered_bipolar_diagnosis_timestamps_df["timestamp"]
            >= filtered_bipolar_diagnosis_timestamps_df["timestamp_first_visit"]
        ]

        filtered_bipolar_diagnosis_timestamps_df["time_from_first_visit"] = (
            filtered_bipolar_diagnosis_timestamps_df["timestamp"]
            - filtered_bipolar_diagnosis_timestamps_df["timestamp_first_visit"]
        )

        timestamps_per_patient = []

        for _, row in tqdm(filtered_bipolar_diagnosis_timestamps_df.iterrows()):
            timestamps = generate_timestamps(
                row["timestamp_first_visit"], row["timestamp"], interval_days=interval_days
            )
            timestamps_per_patient.extend(
                [(row["dw_ek_borger"], timestamp) for timestamp in timestamps]
            )

            result_df = pd.DataFrame(timestamps_per_patient, columns=["patient_id", "timestamp"])

        prediction_times = PredictionTimeFrame(frame=result_df)  # type: ignore
        filtered_bipolar_diagnosis_timestamps.prediction_times.frame = prediction_times  # type: ignore

        return filtered_bipolar_diagnosis_timestamps


if __name__ == "__main__":
    df = BipolarCohortDefiner.get_bipolar_cohort()
