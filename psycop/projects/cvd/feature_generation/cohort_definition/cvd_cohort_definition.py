import polars as pl

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    filter_prediction_times,
)
from psycop.common.feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)
from psycop.projects.cvd.feature_generation.cohort_definition.eligible_prediction_times.single_filters import (
    CVDMinAgeFilter,
    CVDMinDateFilter,
    CVDWashoutMove,
    NoIncidentCVD,
    WithoutPrevalentCVD,
)
from psycop.projects.cvd.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_cvd_indicator,
)


class CVDCohortDefiner(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle:
        unfiltered_prediction_times = pl.from_pandas(
            physical_visits_to_psychiatry(
                timestamps_only=True,
                timestamp_for_output="start",
            ),
        )

        return filter_prediction_times(
            prediction_times=unfiltered_prediction_times,
            filtering_steps=(
                CVDMinDateFilter(),
                CVDMinAgeFilter(),
                WithoutPrevalentCVD(),
                NoIncidentCVD(),
                CVDWashoutMove(),
            ),
            entity_id_col_name="dw_ek_borger",
        )

    @staticmethod
    def get_outcome_timestamps() -> pl.DataFrame:
        return pl.from_pandas(get_first_cvd_indicator())


if __name__ == "__main__":
    bundle = CVDCohortDefiner.get_filtered_prediction_times_bundle()

    df = bundle.prediction_times

    outcome_timestamps = CVDCohortDefiner.get_outcome_timestamps()
