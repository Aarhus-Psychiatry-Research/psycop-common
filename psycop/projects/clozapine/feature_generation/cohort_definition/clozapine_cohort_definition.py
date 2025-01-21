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
from psycop.projects.clozapine.feature_generation.cohort_definition.eligible_prediction_times.single_filters import (
    ClozapineMinAgeFilter,
    ClozapineMinDateFilter,
    ClozapinePrevalentFilter,
    ClozapineSchizophrenia,
    ClozapineWashoutMoveFilter,
)
from psycop.projects.clozapine.feature_generation.cohort_definition.outcome_specification.combine_text_structured_clozapine_outcome import (
    combine_structured_and_text_outcome,
)
from psycop.projects.clozapine.loaders.visits import physical_visits_to_psychiatry_clozapine_2024


@shared_cache().cache()
def clozapine_pred_filtering() -> FilteredPredictionTimeBundle:
    return ClozapineCohortDefiner().get_filtered_prediction_times_bundle()


@shared_cache().cache()
def clozapine_pred_times() -> PredictionTimeFrame:
    return clozapine_pred_filtering().prediction_times


@shared_cache().cache()
def clozapine_outcome_timestamps() -> OutcomeTimestampFrame:
    return ClozapineCohortDefiner().get_outcome_timestamps()


class ClozapineCohortDefiner(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle:
        unfiltered_prediction_times = pl.from_pandas(
            physical_visits_to_psychiatry_clozapine_2024(timestamps_only=True, timestamp_for_output="start")
        )

        result = filter_prediction_times(
            prediction_times=unfiltered_prediction_times.lazy(),
            filtering_steps=(
                ClozapineSchizophrenia(),
                ClozapineMinDateFilter(),
                ClozapineMinAgeFilter(),
                ClozapinePrevalentFilter(),
                ClozapineWashoutMoveFilter(),
            ),
            entity_id_col_name="dw_ek_borger",
        )

        return result

    @staticmethod
    def get_outcome_timestamps() -> OutcomeTimestampFrame:
        return OutcomeTimestampFrame(
            frame=(
                pl.from_pandas(combine_structured_and_text_outcome())
                .with_columns(value=pl.lit(1))
                .select(["dw_ek_borger", "timestamp", "value"])
            )
        )


if __name__ == "__main__":
    filtered_prediction_time_bundle = ClozapineCohortDefiner.get_filtered_prediction_times_bundle()

    for filtering_step in filtered_prediction_time_bundle.filter_steps:
        msg.info(f"Filter step {filtering_step.step_index} {filtering_step.step_name}")
        msg.info(
            f"\tPrediction times: {filtering_step.n_prediction_times_before} - {filtering_step.n_prediction_times_after} = {filtering_step.n_dropped_prediction_times} dropped prediction times"
        )
        msg.info(
            f"\tUnique patients: {filtering_step.n_ids_before} - {filtering_step.n_ids_after} = {filtering_step.n_dropped_ids} dropped ids"
        )
