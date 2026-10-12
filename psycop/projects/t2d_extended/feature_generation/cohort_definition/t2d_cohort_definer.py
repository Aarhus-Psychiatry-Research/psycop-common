import polars as pl
from wasabi import Printer

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    OutcomeTimestampFrame,
    filter_prediction_times,
)
from psycop.common.feature_generation.loaders_2025.demographics import birthdays
from psycop.common.feature_generation.loaders_2025.visits import physical_visits_to_psychiatry_2025
from psycop.common.global_utils.cache import shared_cache
from psycop.common.sequence_models.registry import SequenceRegistry
from psycop.projects.t2d_extended.feature_generation.cohort_definition.eligible_prediction_times.single_filters import (
    NoIncidentDiabetes,
    T2DMinAgeFilter,
    T2DMinDateFilter,
    WithoutPrevalentDiabetes,
)
from psycop.projects.t2d_extended.feature_generation.cohort_definition.outcome_specification.lab_results import (
    get_first_diabetes_lab_result_above_threshold,
)

msg = Printer(timestamp=True)


@shared_cache().cache()
def t2d_pred_times() -> FilteredPredictionTimeBundle:
    return T2DCohortDefiner2025.get_filtered_prediction_times_bundle()


@shared_cache().cache()
def t2d_outcome_timestamps() -> OutcomeTimestampFrame:
    return T2DCohortDefiner2025.get_outcome_timestamps()


@SequenceRegistry.cohorts.register("t2d_extended")
class T2DCohortDefiner2025(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle:
        msg.info("Getting unfiltered prediction times")
        unfiltered_prediction_times = pl.from_pandas(
            physical_visits_to_psychiatry_2025(timestamps_only=True, timestamp_for_output="start")
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
            ),
            entity_id_col_name="dw_ek_borger",
        )

    @staticmethod
    def get_outcome_timestamps() -> OutcomeTimestampFrame:
        return OutcomeTimestampFrame(
            frame=pl.from_pandas(get_first_diabetes_lab_result_above_threshold())
        )


if __name__ == "__main__":
    bundle = T2DCohortDefiner2025.get_filtered_prediction_times_bundle()

    if isinstance(bundle.prediction_times, pl.LazyFrame):
        msg.info("Collecting")
        df = bundle.prediction_times.collect()
        msg.good("Collected")
    else:
        df = bundle.prediction_times
