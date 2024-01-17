import polars as pl

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    OutcomeTimestampFrame,
    filter_prediction_times,
)
from psycop.common.feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)
from psycop.projects.clozapine.feature_generation.cohort_definition.eligible_prediction_times.single_filters import (
    ClozapineMinAgeFilter,
    ClozapineMinDateFilter,
    ClozapinePrevalentFilter,
    ClozapineSchizophrenia,
    ClozapineWashoutMoveFilter,
)
from psycop.projects.clozapine.feature_generation.cohort_definition.outcome_specification.first_clozapine_prescription import (
    get_first_clozapine_prescription,
)


class ClozapineCohortDefiner(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle:
        unfiltered_prediction_times = pl.from_pandas(
            physical_visits_to_psychiatry(
                timestamps_only=True,
                timestamp_for_output="start",
            ),
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
                pl.from_pandas(get_first_clozapine_prescription())
                .with_columns(value=pl.lit(1))
                .select(["dw_ek_borger", "timestamp", "value"])
            ),
        )


if __name__ == "__main__":
    bundle = ClozapineCohortDefiner.get_filtered_prediction_times_bundle()

    df = bundle.prediction_times

    outcome_timestamps = ClozapineCohortDefiner.get_outcome_timestamps()
