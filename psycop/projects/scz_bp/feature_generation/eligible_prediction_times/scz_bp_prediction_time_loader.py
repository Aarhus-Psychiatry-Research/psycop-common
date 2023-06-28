from collections.abc import Iterable

import polars as pl

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimes,
    PredictionTimeFilter,
    StepDelta,
    filter_prediction_times,
)
from psycop.common.feature_generation.loaders.raw.load_visits import ambulatory_visits
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.single_filters import (
    SczBpAddAgeFilter,
    SczBpExcludedByWashinFilter,
    SczBpMaxAgeFilter,
    SczBpMinAgeFilter,
    SczBpMinDateFilter,
    SczBpPrevalentFilter,
    SczBpWashoutMoveFilter,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.first_scz_or_bp_diagnosis import (
    get_first_scz_bp_diagnosis_after_washin,
)


class SczBpCohort(CohortDefiner):
    @staticmethod
    def get_eligible_prediction_times() -> FilteredPredictionTimes:
        # prediction times are right before an ambulatory visit
        prediction_times = pl.from_pandas(
            ambulatory_visits(
                timestamps_only=True,
                timestamp_for_output="start",
                n_rows=None,
                return_value_as_visit_length_days=False,
                shak_code=6600,
                shak_sql_operator="=",
            ),
        ).with_columns(
            pl.col("timestamp") - pl.duration(days=1),
        )

        return filter_prediction_times(
            prediction_times=prediction_times,
            filtering_steps=SczBpCohort._get_filtering_steps(),
        )

    @staticmethod
    def get_outcome_timestamps() -> pl.DataFrame:
        return get_first_scz_bp_diagnosis_after_washin()

    @staticmethod
    def _get_filtering_steps() -> Iterable[PredictionTimeFilter]:
        return (
            SczBpAddAgeFilter(),
            SczBpMinAgeFilter(),
            SczBpMaxAgeFilter(),
            SczBpMinDateFilter(),
            SczBpExcludedByWashinFilter(),
            SczBpWashoutMoveFilter(),
            SczBpPrevalentFilter(),
        )


if __name__ == "__main__":
    filtered_prediction_times = SczBpCohort.get_eligible_prediction_times()
    for stepdelta in filtered_prediction_times.filter_steps:
        print(
            f"{stepdelta.step_name} dropped {stepdelta.n_dropped}, remaining: {stepdelta.n_after}",
        )

    print(f"Remaining: {filtered_prediction_times.prediction_times.shape[0]}")
