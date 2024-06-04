import logging
from collections.abc import Iterable

import polars as pl
from wasabi import msg

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    OutcomeTimestampFrame,
    PredictionTimeFilter,
    filter_prediction_times,
)
from psycop.common.feature_generation.loaders.raw.load_visits import ambulatory_visits
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.single_filters import (
    SczBpMinDateFilter,
    SczBpPrevalentFilter,
    SczBpTimeFromFirstVisitFilter,
    SczBpWashoutMoveFilter,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.first_scz_or_bp_diagnosis import (
    get_first_scz_or_bp_diagnosis,
)

log = logging.getLogger(__name__)


class SczBpCohort(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle:
        # prediction times are the day before an ambulatory visit
        prediction_times = pl.from_pandas(
            ambulatory_visits(
                timestamps_only=True,
                timestamp_for_output="start",
                n_rows=None,
                return_value_as_visit_length_days=False,
                shak_code=6600,
                shak_sql_operator="=",
            )
        ).with_columns(pl.col("timestamp") - pl.duration(days=1))

        filtered_prediction_time_bundle = filter_prediction_times(
            prediction_times=prediction_times.lazy(),
            filtering_steps=SczBpCohort._get_filtering_steps(),
            entity_id_col_name="dw_ek_borger",
        )

        msg.divider("Loaded SczBp eligible prediction times")
        msg.info("N prediction times and ids after filtering:\n")
        for filtering_step in filtered_prediction_time_bundle.filter_steps:
            msg.info(f"Filter step {filtering_step.step_index} {filtering_step.step_name}")
            msg.info(
                f"\tPrediction times: {filtering_step.n_prediction_times_before} - {filtering_step.n_prediction_times_after} = {filtering_step.n_dropped_prediction_times} dropped prediction times"
            )
            msg.info(
                f"\tUnique patients: {filtering_step.n_ids_before} - {filtering_step.n_ids_after} = {filtering_step.n_dropped_ids} dropped ids"
            )

        return filtered_prediction_time_bundle

    @staticmethod
    def get_outcome_timestamps() -> OutcomeTimestampFrame:
        return OutcomeTimestampFrame(
            frame=get_first_scz_or_bp_diagnosis().select("dw_ek_borger", "timestamp", "value")
        )

    @staticmethod
    def _get_filtering_steps() -> Iterable[PredictionTimeFilter]:
        return (SczBpMinDateFilter(), SczBpTimeFromFirstVisitFilter(), SczBpPrevalentFilter(), SczBpWashoutMoveFilter())


if __name__ == "__main__":
    filtered_prediction_times = SczBpCohort.get_filtered_prediction_times_bundle()
    for stepdelta in filtered_prediction_times.filter_steps:
        print(
            f"{stepdelta.step_name} dropped {stepdelta.n_dropped_prediction_times}, remaining: {stepdelta.n_prediction_times_after}"
        )

    print(f"Remaining: {filtered_prediction_times.prediction_times.frame.shape[0]}")

    diag = get_first_scz_or_bp_diagnosis().select("dw_ek_borger", "source")
    pos = filtered_prediction_times.prediction_times.frame.join(diag, on="dw_ek_borger")

    pos.groupby("source").agg(pl.col("dw_ek_borger").unique().len())
