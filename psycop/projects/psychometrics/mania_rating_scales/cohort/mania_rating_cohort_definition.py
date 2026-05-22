import polars as pl
from wasabi import msg

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    OutcomeTimestampFrame,
    PredictionTimeFrame,
    filter_prediction_times,
)
from psycop.common.global_utils.cache import shared_cache
from psycop.projects.psychometrics.loaders.visits import (
    physical_visits_to_psychiatry_psykometri_2025,
)
from psycop.projects.psychometrics.mania_rating_scales.cohort.eligible_prediction_times.single_filters import (
    ManiaRatingBipolarDisorders,
    ManiaRatingWashoutMoveFilter,
    PsychometricsMinAgeFilter,
    PsychometricsMinDateFilter,
)
from psycop.projects.psychometrics.mania_rating_scales.cohort.outcome_specification.mania_rating_score import (
    get_mania_rating_scores,
)


@shared_cache().cache()
def mania_rating_pred_filtering() -> FilteredPredictionTimeBundle:
    return ManiaRatingCohortDefiner().get_filtered_prediction_times_bundle()


@shared_cache().cache()
def mania_rating_pred_times() -> PredictionTimeFrame:
    return mania_rating_pred_filtering().prediction_times


@shared_cache().cache()
def mania_rating_outcome_timestamps() -> OutcomeTimestampFrame:
    return ManiaRatingCohortDefiner().get_outcome_timestamps()


class ManiaRatingCohortDefiner(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle:
        unfiltered_prediction_times = pl.from_pandas(
            physical_visits_to_psychiatry_psykometri_2025(
                timestamps_only=True, timestamp_for_output="start"
            )
        )

        result = filter_prediction_times(
            prediction_times=unfiltered_prediction_times.lazy(),
            filtering_steps=(
                PsychometricsMinAgeFilter(),
                PsychometricsMinDateFilter(),
                ManiaRatingBipolarDisorders(),
                ManiaRatingWashoutMoveFilter(),
            ),
            entity_id_col_name="dw_ek_borger",
        )

        return result

    @staticmethod
    def get_outcome_timestamps() -> OutcomeTimestampFrame:
        # Load all outcome timestamps
        frame = (
            pl.from_pandas(get_mania_rating_scores())
            .with_columns(value=pl.lit(1))
            .select(["dw_ek_borger", "timestamp", "value"])
        )

        return OutcomeTimestampFrame(frame=frame)


if __name__ == "__main__":
    filtered_prediction_time_bundle = (
        ManiaRatingCohortDefiner.get_filtered_prediction_times_bundle()
    )

    outcome_timestamps_bundle = ManiaRatingCohortDefiner.get_outcome_timestamps()

    for filtering_step in filtered_prediction_time_bundle.filter_steps:
        msg.info(f"Filter step {filtering_step.step_index} {filtering_step.step_name}")
        msg.info(
            f"\tPrediction times: {filtering_step.n_prediction_times_before} - {filtering_step.n_prediction_times_after} = {filtering_step.n_dropped_prediction_times} dropped prediction times"
        )
        msg.info(
            f"\tUnique patients: {filtering_step.n_ids_before} - {filtering_step.n_ids_after} = {filtering_step.n_dropped_ids} dropped ids"
        )
