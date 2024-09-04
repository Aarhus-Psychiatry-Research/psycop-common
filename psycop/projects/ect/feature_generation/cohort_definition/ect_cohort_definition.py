import polars as pl
from wasabi import msg
from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    OutcomeTimestampFrame,
    PredictionTimeFrame,
    filter_prediction_times,
)
from psycop.common.feature_generation.loaders.raw.load_visits import admissions
from psycop.common.global_utils.cache import shared_cache
from psycop.projects.ect.feature_generation.cohort_definition.eligible_prediction_times.single_filters import (
    ECTMinAgeFilter,
    ECTMinDateFilter,
    ECTWashoutMove,
    NoIncidentECTWithin3Years,
    NoIncidentF2,
    NoIncidentF6
)
from psycop.projects.ect.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_ect_indicator,
)


@shared_cache().cache()
def ect_pred_filtering() -> FilteredPredictionTimeBundle:
    return ECTCohortDefiner().get_filtered_prediction_times_bundle()


@shared_cache().cache()
def ect_pred_times() -> PredictionTimeFrame:
    return ect_pred_filtering().prediction_times


@shared_cache().cache()
def ect_outcome_timestamps() -> OutcomeTimestampFrame:
    return ECTCohortDefiner().get_outcome_timestamps()


class ECTCohortDefiner(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle:
        # make predictions 7 days after admission to not make predictions
        # for patients in acute need (which would already be known)
        unfiltered_prediction_times = pl.from_pandas(
            admissions(timestamps_only=True, timestamp_for_output="start")
        ).with_columns(pl.col("timestamp") + pl.duration(days=7)) 

        result = filter_prediction_times(
            prediction_times=unfiltered_prediction_times.lazy(),
            filtering_steps=(
                ECTMinDateFilter(),
                ECTMinAgeFilter(),
                NoIncidentECTWithin3Years(),
                ECTWashoutMove(),
                NoIncidentF2(),
                NoIncidentF6()
            ),
            entity_id_col_name="dw_ek_borger",
        )

        return result

    @staticmethod
    def get_outcome_timestamps() -> OutcomeTimestampFrame:
        return OutcomeTimestampFrame(
            frame=(
                pl.from_pandas(get_first_ect_indicator())
                .with_columns(value=pl.lit(1))
                .select(["dw_ek_borger", "timestamp", "value"])
            )
        )


if __name__ == "__main__":
    filtered_prediction_time_bundle = ECTCohortDefiner.get_filtered_prediction_times_bundle()
    
    for filtering_step in filtered_prediction_time_bundle.filter_steps:
        msg.info(f"Filter step {filtering_step.step_index} {filtering_step.step_name}")
        msg.info(
            f"\tPrediction times: {filtering_step.n_prediction_times_before} - {filtering_step.n_prediction_times_after} = {filtering_step.n_dropped_prediction_times} dropped prediction times"
        )
        msg.info(
            f"\tUnique patients: {filtering_step.n_ids_before} - {filtering_step.n_ids_after} = {filtering_step.n_dropped_ids} dropped ids"
        )
    # cohort = pl.read_parquet(
    #     "E:/shared_resources/ect/feature_set/flattened_datasets/ect_feature_set/ect_feature_set.parquet"
    # )
