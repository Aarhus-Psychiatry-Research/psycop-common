import polars as pl

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    OutcomeTimestampFrame,
    filter_prediction_times,
)
#get timestamps to somatic admission. These timestampts can be used for filtering and for outcome
from psycop.projects.AcuteSomaticAdmission.CohortDefinition.get_somatic_emergency_visits import (
    get_contacts_to_somatic_emergency,
)
#get timestamps for outpatient visits
from psycop.projects.AcuteSomaticAdmission.CohortDefinition.get_psychiatric_outpatient_visits import (
    get_outpatient_visits_to_psychiatry,
)
from psycop.projects.AcuteSomaticAdmission.CohortDefinition.single_filters import (
    SomaticAdmissionMinAgeFilter,
    SomaticAdmissionMinDateFilter,
    SomaticAdmissionWashoutMove,
    SomaticAdmissionWashoutPriorSomaticAdmission,
)

class SomaticAdmissionCohortDefiner(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle(
        washout_on_prior_somatic_contacts: bool = True,
    ) -> FilteredPredictionTimeBundle:
        unfiltered_prediction_times = pl.from_pandas(get_outpatient_visits_to_psychiatry()).lazy()

        if washout_on_prior_somatic_contacts:
            return filter_prediction_times(
                prediction_times=unfiltered_prediction_times,
                filtering_steps=(
                    SomaticAdmissionMinDateFilter(),
                    SomaticAdmissionMinAgeFilter(),
                    SomaticAdmissionWashoutMove(),
                    SomaticAdmissionWashoutPriorSomaticAdmission(),
                ),
                entity_id_col_name="dw_ek_borger",
            )

        return filter_prediction_times(
            prediction_times=unfiltered_prediction_times,
            filtering_steps=(
                    SomaticAdmissionMinDateFilter(),
                    SomaticAdmissionMinAgeFilter(),
            ),
            entity_id_col_name="dw_ek_borger",
        )

    @staticmethod
    def get_outcome_timestamps() -> OutcomeTimestampFrame:
        return OutcomeTimestampFrame(frame=pl.from_pandas(get_contacts_to_somatic_emergency()))


if __name__ == "__main__":
    bundle = SomaticAdmissionCohortDefiner.get_filtered_prediction_times_bundle()

    bundle_no_washout = (
        SomaticAdmissionCohortDefiner.get_filtered_prediction_times_bundle(
            washout_on_prior_somatic_contacts=False
        )
    )

    df = bundle.prediction_times.frame.to_pandas()

    df_no_washout = bundle_no_washout.prediction_times.frame.to_pandas()

    outcome_timestamps = SomaticAdmissionCohortDefiner.get_outcome_timestamps()
