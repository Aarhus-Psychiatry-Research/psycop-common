import polars as pl

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    filter_prediction_times,
    OutcomeTimestampFrame
)
from psycop.common.feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)

from psycop.projects.cancer.feature_generation.cohort_definition.eligible_prediction_times.single_filters import (
    CancerMinAgeFilter,
    CancerMinDateFilter,
    CancerPrevalentFilter,
    CancerWashoutMoveFilter,
)
from psycop.projects.cancer.feature_generation.cohort_definition.outcome_specification.first_cancer_diagnosis import (
    get_first_cancer_diagnosis,
)


class CancerCohortDefiner(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle:
        unfiltered_prediction_times = pl.from_pandas(
            physical_visits_to_psychiatry(
                timestamps_only=True,
                timestamp_for_output="start",
            ),
        )

        return filter_prediction_times(
            prediction_times=unfiltered_prediction_times.lazy(),
            filtering_steps=(
                CancerMinDateFilter(),
                CancerMinAgeFilter(),
                CancerWashoutMoveFilter(),
                CancerPrevalentFilter(),
            ),
            entity_id_col_name="dw_ek_borger",
        )

    @staticmethod
    def get_outcome_timestamps() -> OutcomeTimestampFrame:
        return OutcomeTimestampFrame(frame=pl.from_pandas(get_first_cancer_diagnosis()))


if __name__ == "__main__":
    bundle = CancerCohortDefiner.get_filtered_prediction_times_bundle()

    df = bundle.prediction_times
