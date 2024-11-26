import polars as pl
from wasabi import Printer

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    OutcomeTimestampFrame,
    filter_prediction_times,
)
from psycop.common.global_utils.cache import shared_cache
from psycop.common.sequence_models.registry import SequenceRegistry
from psycop.projects.uti.feature_generation.cohort_definition.eligible_prediction_times.functions import (
    explode_admissions,
    load_admissions_discharge_timestamps,
    preprocess_readmissions,
)
from psycop.projects.uti.feature_generation.cohort_definition.eligible_prediction_times.single_filters import (
    UTIAdmissionFilter,
    UTIAdmissionTypeFilter,
    UTIExcludeFirstDayFilter,
    UTIMinAgeFilter,
    UTIMinDateFilter,
    UTIShakCodeFilter,
)
from psycop.projects.uti.feature_generation.outcome_definition.uti_outcomes import uti_outcomes

msg = Printer(timestamp=True)


@shared_cache().cache()
def uti_pred_times() -> FilteredPredictionTimeBundle:
    return UTICohortDefiner.get_filtered_prediction_times_bundle()


@SequenceRegistry.cohorts.register("uti")
class UTICohortDefiner(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle:
        unfiltered_prediction_times = pl.LazyFrame(
            pl.from_pandas(
                load_admissions_discharge_timestamps()[0:100].rename(
                    columns={"datotid_start": "timestamp"}
                )
            )
        )

        filtered_prediction_times = filter_prediction_times(
            prediction_times=unfiltered_prediction_times,
            filtering_steps=(UTIAdmissionTypeFilter(), UTIShakCodeFilter(), UTIMinAgeFilter()),
            entity_id_col_name="dw_ek_borger",
        ).prediction_times.frame.select(  # type: ignore
            pl.col(["dw_ek_borger", "timestamp", "datotid_slut", "shakkode_ansvarlig"])
        )

        filtered_prediction_times = preprocess_readmissions(df=filtered_prediction_times)

        filtered_prediction_times = pl.LazyFrame(
            filter_prediction_times(
                prediction_times=filtered_prediction_times,
                filtering_steps=[UTIMinDateFilter()],
                entity_id_col_name="dw_ek_borger",
            ).prediction_times.frame
        )

        filtered_cohort = pl.LazyFrame(
            filter_prediction_times(
                prediction_times=filtered_prediction_times,
                filtering_steps=[UTIAdmissionFilter()],
                entity_id_col_name="dw_ek_borger",
            ).prediction_times.frame
        )

        exploded_cohort = explode_admissions(filtered_cohort)

        return filter_prediction_times(
            prediction_times=exploded_cohort,  # type: ignore
            filtering_steps=[UTIExcludeFirstDayFilter()],
            entity_id_col_name="dw_ek_borger",
        )

    @staticmethod
    def get_outcome_timestamps() -> OutcomeTimestampFrame:
        return OutcomeTimestampFrame(frame=pl.from_pandas(uti_outcomes()))


if __name__ == "__main__":
    bundle = UTICohortDefiner.get_filtered_prediction_times_bundle()

    if isinstance(bundle.prediction_times, pl.LazyFrame):
        msg.info("Collecting")
        df = bundle.prediction_times.collect()
        msg.good("Collected")
    else:
        df = bundle.prediction_times
