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
    ECTMaxDateFilter,
    ECTMinAgeFilter,
    ECTMinDateFilter,
    ECTWashoutMove,
    NoIncidentECTWithin3Years,
    NoIncidentF2,
    NoIncidentF6,
)
from psycop.projects.ect.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_ect_indicator,
)
from psycop.projects.restraint.cohort.utils.functions import preprocess_readmissions


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

        # Load start and end admission timestamps
        admissions_start = pl.from_pandas(
            admissions(
                shak_code=6600,
                shak_sql_operator="=",
                timestamps_only=True,
                timestamp_for_output="start",
            )
        )
        admissions_end = pl.from_pandas(
            admissions(
                shak_code=6600,
                shak_sql_operator="=",
                timestamps_only=True,
                timestamp_for_output="end",
                remove_na_timestamp_rows=False,
            )
        )

        # Merge
        unfiltered_prediction_times = admissions_start.with_columns(
            admissions_end["timestamp"].alias("datotid_slut")
        )

        # Add shakkode column
        unfiltered_prediction_times = unfiltered_prediction_times.with_columns(
            pl.lit(6600).alias("shakkode_ansvarlig")
        )

        unfiltered_prediction_times = unfiltered_prediction_times.rename(
            {"timestamp": "datotid_start"}
        )

        # Concatenate admissions where a new admission is started within 8 hours following discharge
        unfiltered_prediction_times = preprocess_readmissions(df=unfiltered_prediction_times)

        # Create prediction timestamps 7 days after admission time
        unfiltered_prediction_times = unfiltered_prediction_times.collect().with_columns(
            pl.col("datotid_start") + pl.duration(days=7)
        )

        # Remove rows where prediction timestamp is after discharge
        unfiltered_prediction_times = unfiltered_prediction_times.filter(
            pl.col("datotid_slut").is_not_null()
            & (pl.col("datotid_start") <= pl.col("datotid_slut"))
        )

        # Rename timestamp column
        unfiltered_prediction_times = unfiltered_prediction_times[
            ["dw_ek_borger", "datotid_start"]
        ].rename({"datotid_start": "timestamp"})

        result = filter_prediction_times(
            prediction_times=unfiltered_prediction_times.lazy(),
            filtering_steps=(
                ECTMinDateFilter(),
                ECTMaxDateFilter(),
                ECTMinAgeFilter(),
                NoIncidentECTWithin3Years(),
                ECTWashoutMove(),
                NoIncidentF2(),
                NoIncidentF6(),
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
    # cohort = pl.read_parquet( # noqa: ERA001
    #     "E:/shared_resources/ect/feature_set/flattened_datasets/ect_feature_set/ect_feature_set.parquet" # noqa: ERA001
    # ) noqa: ERA001
