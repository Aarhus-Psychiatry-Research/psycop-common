import polars as pl

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    filter_prediction_times,
)
from psycop.projects.forced_admission_inpatient.cohort.extract_admissions_and_visits.get_forced_admissions import (
    forced_admissions_onset_timestamps,
)
from psycop.projects.forced_admission_inpatient.cohort.extract_admissions_and_visits.get_inpatient_admissions_to_psychiatry import (
    admissions_discharge_timestamps,
)
from psycop.projects.forced_admission_inpatient.cohort.prediction_timestamp_filters.single_filters import (
    ForcedAdmissionsInpatientMinAgeFilter,
    ForcedAdmissionsInpatientMinDateFilter,
    ForcedAdmissionsInpatientWashoutMove,
    ForcedAdmissionsInpatientWashoutPriorForcedAdmission,
)


class ForcedAdmissionsInpatientCohortDefiner(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle:
        unfiltered_prediction_times = pl.from_pandas(
            admissions_discharge_timestamps(),
        )

        return filter_prediction_times(
            prediction_times=unfiltered_prediction_times,
            filtering_steps=(
                ForcedAdmissionsInpatientMinDateFilter(),
                ForcedAdmissionsInpatientMinAgeFilter(),
                ForcedAdmissionsInpatientWashoutMove(),
                ForcedAdmissionsInpatientWashoutPriorForcedAdmission(),
            ),
            entity_id_col_name="dw_ek_borger",
        )

    @staticmethod
    def get_outcome_timestamps() -> pl.DataFrame:
        return pl.from_pandas(forced_admissions_onset_timestamps())


if __name__ == "__main__":
    bundle = (
        ForcedAdmissionsInpatientCohortDefiner.get_filtered_prediction_times_bundle()
    )

    df = bundle.prediction_times

    outcome_timestamps = ForcedAdmissionsInpatientCohortDefiner.get_outcome_timestamps()
