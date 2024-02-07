import polars as pl

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    OutcomeTimestampFrame,
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
    def get_filtered_prediction_times_bundle(
        washout_on_prior_forced_admissions: bool = True,
    ) -> FilteredPredictionTimeBundle:
        unfiltered_prediction_times = pl.from_pandas(admissions_discharge_timestamps()).lazy()

        if washout_on_prior_forced_admissions:
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

        return filter_prediction_times(
            prediction_times=unfiltered_prediction_times,
            filtering_steps=(
                ForcedAdmissionsInpatientMinDateFilter(),
                ForcedAdmissionsInpatientMinAgeFilter(),
            ),
            entity_id_col_name="dw_ek_borger",
        )

    @staticmethod
    def get_outcome_timestamps() -> OutcomeTimestampFrame:
        return OutcomeTimestampFrame(frame=pl.from_pandas(forced_admissions_onset_timestamps()))


if __name__ == "__main__":
    bundle = ForcedAdmissionsInpatientCohortDefiner.get_filtered_prediction_times_bundle()

    bundle_no_washout = ForcedAdmissionsInpatientCohortDefiner.get_filtered_prediction_times_bundle(
        washout_on_prior_forced_admissions=False
    )

    df = bundle.prediction_times.frame.to_pandas()

    df_no_washout = bundle_no_washout.prediction_times.frame.to_pandas()

    outcome_timestamps = ForcedAdmissionsInpatientCohortDefiner.get_outcome_timestamps()
