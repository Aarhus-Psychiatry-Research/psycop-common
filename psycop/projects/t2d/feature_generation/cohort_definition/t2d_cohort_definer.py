import polars as pl

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    filter_prediction_times,
)
from psycop.common.feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)
from psycop.projects.t2d.feature_generation.cohort_definition.eligible_prediction_times.single_filters import (
    NoIncidentDiabetes,
    T2DMinAgeFilter,
    T2DMinDateFilter,
    T2DWashoutMove,
    WithoutPrevalentDiabetes,
)
from psycop.projects.t2d.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_diabetes_indicator,
)

from .....common.feature_generation.loaders.raw.load_demographic import birthdays


class T2DCohortDefiner(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle:
        unfiltered_prediction_times = pl.from_pandas(
            physical_visits_to_psychiatry(
                timestamps_only=True,
                timestamp_for_output="start",
            ),
        ).lazy()

        return filter_prediction_times(
            prediction_times=unfiltered_prediction_times,
            get_counts=False,
            filtering_steps=(
                T2DMinDateFilter(),
                T2DMinAgeFilter(birthday_df=pl.from_pandas(birthdays()).lazy()),
                WithoutPrevalentDiabetes(),
                NoIncidentDiabetes(),
                T2DWashoutMove(),
            ),
            entity_id_col_name="dw_ek_borger",
        )

    @staticmethod
    def get_outcome_timestamps() -> pl.DataFrame:
        return pl.from_pandas(get_first_diabetes_indicator())


if __name__ == "__main__":
    bundle = T2DCohortDefiner.get_filtered_prediction_times_bundle()

    df = bundle.prediction_times
