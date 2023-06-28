import polars as pl

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimes,
    filter_prediction_times,
)
from psycop.projects.t2d.feature_generation.eligible_prediction_times.loader import (
    get_unfiltered_t2d_prediction_times_as_polars,
)
from psycop.projects.t2d.feature_generation.eligible_prediction_times.single_filters import (
    NoIncidentDiabetes,
    T2DMinAgeFilter,
    T2DMinDateFilter,
    T2DWashoutMove,
    WithoutPrevalentDiabetes,
)
from psycop.projects.t2d.feature_generation.outcome_specification.combined import (
    get_first_diabetes_indicator,
)


class T2DCohortDefiner(CohortDefiner):
    @staticmethod
    def get_eligible_prediction_times() -> FilteredPredictionTimes:
        unfiltered_prediction_times = get_unfiltered_t2d_prediction_times_as_polars()

        return filter_prediction_times(
            prediction_times=unfiltered_prediction_times,
            filtering_steps=(
                T2DMinDateFilter(),
                T2DMinAgeFilter(),
                WithoutPrevalentDiabetes(),
                NoIncidentDiabetes(),
                T2DWashoutMove(),
            ),
        )

    @staticmethod
    def get_outcome_timestamps() -> pl.DataFrame:
        return pl.from_pandas(get_first_diabetes_indicator())
