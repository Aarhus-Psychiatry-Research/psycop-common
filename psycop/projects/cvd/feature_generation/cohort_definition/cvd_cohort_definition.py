import polars as pl

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    OutcomeTimestampFrame,
    PredictionTimeFrame,
    filter_prediction_times,
)
from psycop.common.feature_generation.loaders.raw.load_visits import physical_visits_to_psychiatry
from psycop.common.global_utils.cache import shared_cache
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


@shared_cache.cache()
def cvd_pred_filtering() -> FilteredPredictionTimeBundle:
    return CVDCohortDefiner().get_filtered_prediction_times_bundle()


@shared_cache.cache()
def cvd_pred_times() -> PredictionTimeFrame:
    return cvd_pred_filtering().prediction_times


@shared_cache.cache()
def cvd_outcome_timestamps() -> OutcomeTimestampFrame:
    return CVDCohortDefiner().get_outcome_timestamps()


class CVDCohortDefiner(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle:
        unfiltered_prediction_times = pl.from_pandas(
            physical_visits_to_psychiatry(timestamps_only=True, timestamp_for_output="start")
        )

        result = filter_prediction_times(
            prediction_times=unfiltered_prediction_times.lazy(),
            filtering_steps=(
                CVDMinDateFilter(),
                CVDMinAgeFilter(),
                WithoutPrevalentCVD(),
                NoIncidentCVD(),
                CVDWashoutMove(),
            ),
            entity_id_col_name="dw_ek_borger",
        )

        return result

    @staticmethod
    def get_outcome_timestamps() -> OutcomeTimestampFrame:
        return OutcomeTimestampFrame(
            frame=(
                pl.from_pandas(get_first_cvd_indicator())
                .with_columns(value=pl.lit(1))
                .select(["dw_ek_borger", "timestamp", "value"])
            )
        )


if __name__ == "__main__":
    cohort = pl.read_parquet(
        "E:/shared_resources/cvd/feature_set/flattened_datasets/cvd_feature_set/cvd_feature_set.parquet"
    )
