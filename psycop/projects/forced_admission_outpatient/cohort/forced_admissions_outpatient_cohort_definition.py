import polars as pl

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    OutcomeTimestampFrame,
    filter_prediction_times,
)
from psycop.projects.forced_admission_outpatient.cohort.extract_admissions_and_visits.get_forced_admissions import (
    forced_admissions_onset_timestamps,
)
from psycop.projects.forced_admission_outpatient.cohort.extract_admissions_and_visits.get_outpatient_visits_to_psychiatry import (
    outpatient_visits_timestamps,
)
from psycop.projects.forced_admission_outpatient.cohort.prediction_timestamp_filters.single_filters import (
    ForcedAdmissionsOutpatientMinAgeFilter,
    ForcedAdmissionsOutpatientMinDateFilter,
    ForcedAdmissionsOutpatientWashoutMove,
    ForcedAdmissionsOutpatientWashoutPriorForcedAdmission,
)


class ForcedAdmissionsOutpatientCohortDefiner(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle(
        washout_on_prior_forced_admissions: bool = True,
    ) -> FilteredPredictionTimeBundle:
        unfiltered_prediction_times = pl.from_pandas(outpatient_visits_timestamps()).lazy()

        if washout_on_prior_forced_admissions:
            return filter_prediction_times(
                prediction_times=unfiltered_prediction_times,
                filtering_steps=(
                    ForcedAdmissionsOutpatientMinDateFilter(),
                    ForcedAdmissionsOutpatientMinAgeFilter(),
                    ForcedAdmissionsOutpatientWashoutMove(),
                    ForcedAdmissionsOutpatientWashoutPriorForcedAdmission(),
                ),
                entity_id_col_name="dw_ek_borger",
            )

        return filter_prediction_times(
            prediction_times=unfiltered_prediction_times,
            filtering_steps=(
                ForcedAdmissionsOutpatientMinDateFilter(),
                ForcedAdmissionsOutpatientMinAgeFilter(),
            ),
            entity_id_col_name="dw_ek_borger",
        )

    @staticmethod
    def get_outcome_timestamps() -> OutcomeTimestampFrame:
        return OutcomeTimestampFrame(frame=pl.from_pandas(forced_admissions_onset_timestamps()))


if __name__ == "__main__":
    bundle = ForcedAdmissionsOutpatientCohortDefiner.get_filtered_prediction_times_bundle()

    bundle_no_washout = ForcedAdmissionsOutpatientCohortDefiner.get_filtered_prediction_times_bundle(
        washout_on_prior_forced_admissions=False
    )

    df = bundle.prediction_times.frame.to_pandas()

    df_no_washout = bundle_no_washout.prediction_times.frame.to_pandas()

    outcome_timestamps = ForcedAdmissionsOutpatientCohortDefiner.get_outcome_timestamps()
