import polars as pl
from wasabi import Printer

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    OutcomeTimestampFrame,
    filter_prediction_times,
)
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays
from psycop.common.feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)
from psycop.common.sequence_models.registry import Registry
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

msg = Printer(timestamp=True)


@Registry.cohorts.register("t2d")
class T2DCohortDefiner(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle:
        msg.info("Getting unfiltered prediction times")
        unfiltered_prediction_times = pl.from_pandas(
            physical_visits_to_psychiatry(
                timestamps_only=True,
                timestamp_for_output="start",
            ),
        ).lazy()

        msg.info("Filtering prediction times")
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
    def get_outcome_timestamps() -> OutcomeTimestampFrame:
        return OutcomeTimestampFrame(frame=pl.from_pandas(get_first_diabetes_indicator()))


if __name__ == "__main__":
    bundle = T2DCohortDefiner.get_filtered_prediction_times_bundle()

    if isinstance(bundle.prediction_times, pl.LazyFrame):
        msg.info("Collecting")
        df = bundle.prediction_times.collect()
        msg.good("Collected")
    else:
        df = bundle.prediction_times
