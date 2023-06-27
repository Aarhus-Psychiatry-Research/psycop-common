from typing import Callable

import polars as pl

from psycop.common.cohort_definition import Cohort, FilteredCohort, StepDelta
from psycop.common.feature_generation.loaders.raw.load_visits import ambulatory_visits
from psycop.projects.scz_bp.feature_generation.outcome_specification.add_time_from_first_visit import add_time_from_first_visit
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.single_filters import (
    add_age,
    excluded_by_washin,
    max_age,
    min_age,
    min_date,
    washout_move,
    without_prevalent_scz_or_bp,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.first_scz_or_bp_diagnosis import (
    get_first_scz_bp_diagnosis_after_washin,
)


class SczBpCohort(Cohort):
    def __init__(self):
        self.stepdeltas: list[StepDelta] = []

    def get_eligible_prediction_times(self) -> FilteredCohort:
        prediction_times = self._get_eligible_prediction_times()
        return FilteredCohort(cohort=prediction_times, filter_steps=self.stepdeltas)

    def get_outcome_timestamps(self) -> pl.DataFrame:
        return get_first_scz_bp_diagnosis_after_washin()

    def _get_eligible_prediction_times(self) -> pl.DataFrame:
        df = pl.from_pandas(
            ambulatory_visits(
                timestamps_only=True,
                timestamp_for_output="start",
                n_rows=None,
                return_value_as_visit_length_days=False,
                shak_code=6600,
                shak_sql_operator="=",
            ),
        )
        for (
            filtering_step_name,
            filtering_step_fn,
        ) in self._get_filtering_steps().items():
            n_before = df.shape[0]
            df = filtering_step_fn(df)
            self.stepdeltas.append(
                StepDelta(
                    step_name=filtering_step_name,
                    n_before=n_before,
                    n_after=df.shape[0],
                ),
            )
        return df

    def _get_filtering_steps(self) -> dict[str, Callable]:
        return {
            "add_age": add_age,
            "min_age": min_age,
            "max_age": max_age,
            "min_date": min_date,
            "washin": excluded_by_washin,
            "washout_move": washout_move,
            "prevalent_scz_or_bp" : without_prevalent_scz_or_bp
        }


if __name__ == "__main__":
    cohort = SczBpCohort()
    df = cohort.get_eligible_prediction_times().cohort
    for stepdelta in cohort.stepdeltas:
        print(
            f"{stepdelta.step_name} dropped {stepdelta.n_dropped}, remaining: {stepdelta.n_after}",
        )

    print(f"Remaining: {df.shape[0]}")
