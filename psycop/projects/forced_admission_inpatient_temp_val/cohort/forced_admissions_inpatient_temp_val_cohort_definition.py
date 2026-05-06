import polars as pl

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    OutcomeTimestampFrame,
    PredictionTimeFrame,
    filter_prediction_times,
)
from psycop.common.global_utils.cache import shared_cache
from psycop.projects.forced_admission_inpatient_temp_val.cohort.extract_admissions_and_visits.get_forced_admissions import (
    forced_admissions_onset_timestamps_2025,
)
from psycop.projects.forced_admission_inpatient_temp_val.cohort.extract_admissions_and_visits.get_inpatient_admissions_to_psychiatry import (
    admissions_discharge_timestamps_2025,
)
from psycop.projects.forced_admission_inpatient_temp_val.cohort.prediction_timestamp_filters.single_filters import (
    ForcedAdmissionsInpatientMinAgeFilter,
    ForcedAdmissionsInpatientMinDateFilter,
    ForcedAdmissionsInpatientWashoutMove,
    ForcedAdmissionsInpatientWashoutPriorForcedAdmission,
)


@shared_cache().cache()
def forced_adm_temp_val_pred_filtering() -> FilteredPredictionTimeBundle:
    return ForcedAdmissionsInpatientTempValCohortDefiner().get_filtered_prediction_times_bundle()


@shared_cache().cache()
def forced_adm_temp_val_pred_times() -> PredictionTimeFrame:
    return forced_adm_temp_val_pred_filtering().prediction_times


class ForcedAdmissionsInpatientTempValCohortDefiner(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle(
        washout_on_prior_forced_admissions: bool = True,
    ) -> FilteredPredictionTimeBundle:
        unfiltered_prediction_times = pl.from_pandas(admissions_discharge_timestamps_2025()).lazy()

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
        return OutcomeTimestampFrame(
            frame=pl.from_pandas(forced_admissions_onset_timestamps_2025())
        )


if __name__ == "__main__":
    bundle = ForcedAdmissionsInpatientTempValCohortDefiner.get_filtered_prediction_times_bundle()

    bundle_no_washout = (
        ForcedAdmissionsInpatientTempValCohortDefiner.get_filtered_prediction_times_bundle(
            washout_on_prior_forced_admissions=False
        )
    )

    df = bundle.prediction_times.frame.to_pandas()

    df_no_washout = bundle_no_washout.prediction_times.frame.to_pandas()

    outcome_timestamps = ForcedAdmissionsInpatientTempValCohortDefiner.get_outcome_timestamps()
