import polars as pl
from wasabi import Printer

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    OutcomeTimestampFrame,
    PredictionTimeFrame,
    filter_prediction_times,
)
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays

# TODO: new loader?
from psycop.common.feature_generation.loaders.raw.load_visits import physical_visits_to_psychiatry

# TODO: new loader?
from psycop.common.global_utils.cache import shared_cache
from psycop.common.sequence_models.registry import SequenceRegistry
from psycop.projects.t2d_bigdata.feature_generation.cohort_definition.eligible_prediction_times.single_filters import (
    NoIncidentDiabetes,
    T2DMinAgeFilter,
    T2DMinDateFilter,
    T2DWashoutMove,
    WithoutPrevalentDiabetes,
)
from psycop.projects.t2d_bigdata.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_diabetes_lab_result_above_threshold,
)

msg = Printer(timestamp=True)


@shared_cache().cache()
def t2d_bigdata_pred_filtering() -> FilteredPredictionTimeBundle:
    return T2DBigDataCohortDefiner().get_filtered_prediction_times_bundle()  # TODO:


@shared_cache().cache()
def t2d_bigdata_pred_times() -> PredictionTimeFrame:
    return t2d_bigdata_pred_filtering().prediction_times  # TODO:


@shared_cache().cache()
def t2d_bigdata_outcome_timestamps() -> OutcomeTimestampFrame:
    return T2DBigDataCohortDefiner.get_outcome_timestamps()


@SequenceRegistry.cohorts.register("t2d_bigdata")
class T2DBigDataCohortDefiner(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle:
        msg.info("Getting unfiltered prediction times")
        unfiltered_prediction_times = pl.from_pandas(
            physical_visits_to_psychiatry(timestamps_only=True, timestamp_for_output="start")
        )

        msg.info("Filtering prediction times")
        result = filter_prediction_times(
            prediction_times=unfiltered_prediction_times.lazy(),
            filtering_steps=(
                T2DMinDateFilter(),
                T2DMinAgeFilter(birthday_df=pl.from_pandas(birthdays()).lazy()),
                WithoutPrevalentDiabetes(),
                NoIncidentDiabetes(),
                T2DWashoutMove(),
            ),
            entity_id_col_name="dw_ek_borger",
        )

        return result

    @staticmethod
    def get_outcome_timestamps() -> OutcomeTimestampFrame:
        return OutcomeTimestampFrame(
            frame=pl.from_pandas(get_first_diabetes_lab_result_above_threshold())
        )


if __name__ == "__main__":
    bundle = T2DBigDataCohortDefiner.get_filtered_prediction_times_bundle()

    if isinstance(bundle.prediction_times, pl.LazyFrame):
        msg.info("Collecting")
        df = bundle.prediction_times.collect()
        msg.good("Collected")
    else:
        df = bundle.prediction_times
