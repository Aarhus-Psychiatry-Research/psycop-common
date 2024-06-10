import pandas as pd
import polars as pl
from tqdm import tqdm

from psycop.common.cohort_definition import (
    CohortDefiner,
    PredictionTimeFrame,
    filter_prediction_times,
)
from psycop.common.feature_generation.loaders.raw.load_visits import (
    get_time_of_last_visit_to_psychiatry,
)
from psycop.projects.bipolar.cohort_definition.diagnosis_specification.first_bipolar_diagnosis import (
    get_first_bipolar_diagnosis,
)
from psycop.projects.bipolar.cohort_definition.diagnosis_specification.first_depressive_disorder_diagnosis import (
    get_first_depressive_disorders_diagnosis,
)
from psycop.projects.bipolar.cohort_definition.eligible_data.single_filters import (
    BipolarMinAgeFilter,
    BipolarMinDateFilter,
    BipolarPatientsWithF20F25Filter,
    BipolarPatientsWithF32F38Filter,
    BipolarWashoutMove,
    DepressiveDisorderPatientsWithF31Filter,
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
    def get_bipolar_prediction_times(interval_days: int = 30) -> PredictionTimeFrame:
        # Process prediction times for patients with bipolar disorder (and previous depressive disorder)
        bipolar_diagnosis_timestamps = pl.from_pandas(get_first_bipolar_diagnosis())

        filtered_bipolar_diagnosis_timestamps = filter_prediction_times(
            prediction_times=bipolar_diagnosis_timestamps.lazy(),
            filtering_steps=(
                BipolarMinDateFilter(),
                BipolarMinAgeFilter(),
                BipolarWashoutMove(),
                BipolarPatientsWithF20F25Filter(),
                BipolarPatientsWithF32F38Filter(),
            ),
            entity_id_col_name="dw_ek_borger",
        )

        filtered_bipolar_diagnosis_timestamps_df = pd.DataFrame(
            filtered_bipolar_diagnosis_timestamps.prediction_times.frame.to_pandas()
        )

        first_depressive_disorder_diagnosis = pd.DataFrame(
            get_first_depressive_disorders_diagnosis()
        )

        filtered_bipolar_diagnosis_timestamps_df = filtered_bipolar_diagnosis_timestamps_df.merge(
            first_depressive_disorder_diagnosis,
            on="dw_ek_borger",
            how="left",
            suffixes=(None, "_start"),
        )

        filtered_bipolar_diagnosis_timestamps_df = filtered_bipolar_diagnosis_timestamps_df.dropna(
            subset=["timestamp_first_diagnosis"]
        )

        filtered_bipolar_diagnosis_timestamps_df = filtered_bipolar_diagnosis_timestamps_df[
            filtered_bipolar_diagnosis_timestamps_df["timestamp"]
            >= filtered_bipolar_diagnosis_timestamps_df["timestamp_first_diagnosis"]
        ]

        filtered_bipolar_diagnosis_timestamps_df["timespan_days"] = (
            filtered_bipolar_diagnosis_timestamps_df["timestamp"]
            - filtered_bipolar_diagnosis_timestamps_df["timestamp_start"]
        )

        filtered_bipolar_diagnosis_timestamps_df = filtered_bipolar_diagnosis_timestamps_df.rename(
            columns={"timestamp": "timestamp_end"}
        )

        # Process prediction times for patients with depressive disorder (and no bipolar disorder)
        depressive_disorder_diagnosis_timestamps = pl.from_pandas(
            get_first_depressive_disorders_diagnosis()
        )

        filtered_depressive_disorder_diagnosis_timestamps = filter_prediction_times(
            prediction_times=depressive_disorder_diagnosis_timestamps.lazy(),
            filtering_steps=(
                BipolarMinDateFilter(),
                BipolarMinAgeFilter(),
                BipolarWashoutMove(),
                DepressiveDisorderPatientsWithF31Filter(),
            ),
            entity_id_col_name="dw_ek_borger",
        )

        filtered_depressive_disorder_diagnosis_timestamps_df = pd.DataFrame(
            filtered_depressive_disorder_diagnosis_timestamps.prediction_times.frame.to_pandas()
        )
        last_visits_to_psychiatry = pd.DataFrame(get_time_of_last_visit_to_psychiatry().to_pandas())

        filtered_depressive_disorder_diagnosis_timestamps_df = (
            filtered_depressive_disorder_diagnosis_timestamps_df.merge(
                last_visits_to_psychiatry, on="dw_ek_borger", how="left", suffixes=(None, "_end")
            )
        )

        filtered_depressive_disorder_diagnosis_timestamps_df["timespan_days"] = (
            filtered_depressive_disorder_diagnosis_timestamps_df["timestamp"]
            - filtered_depressive_disorder_diagnosis_timestamps_df["timestamp_end"]
        )

        filtered_depressive_disorder_diagnosis_timestamps_df = (
            filtered_depressive_disorder_diagnosis_timestamps_df.rename(
                columns={"timestamp": "timestamp_start"}
            )
        )

        # Bind the two dataframes
        filtered_prediction_times = pd.concat(
            [
                filtered_bipolar_diagnosis_timestamps_df,
                filtered_depressive_disorder_diagnosis_timestamps_df,
            ]
        )

        timestamps_per_patient = []

        for _, row in tqdm(filtered_prediction_times.iterrows()):
            timestamps = generate_timestamps(
                row["timestamp_start"], row["timestamp_end"], interval_days=interval_days
            )
            timestamps_per_patient.extend(
                [(row["dw_ek_borger"], timestamp) for timestamp in timestamps]
            )

            result_df = pd.DataFrame(timestamps_per_patient, columns=["dw_ek_borger", "timestamp"])

        prediction_times = PredictionTimeFrame(frame=pl.DataFrame(result_df))  # type: ignore

        return prediction_times


if __name__ == "__main__":
    df = BipolarCohortDefiner.get_bipolar_prediction_times()
