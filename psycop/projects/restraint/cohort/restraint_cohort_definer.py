import polars as pl

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    OutcomeTimestampFrame,
    filter_prediction_times,
)
from psycop.projects.restraint.cohort.utils.filters import (
    RestraintAdmissionFilter,
    RestraintAdmissionTypeFilter,
    RestraintCoercionTypeFilter,
    RestraintDoubleAdmissionFilter,
    RestraintExcludeDaysFollowingCoercionFilter,
    RestraintExcludeFirstDayFilter,
    RestraintForcedAdmissionFilter,
    RestraintMinAgeFilter,
    RestraintMinDateFilter,
    RestraintShakCodeFilter,
    RestraintTreatmentUnitFilter,
    RestraintWashoutFilter,
    RestraintWithinAdmissionsFilter,
)
from psycop.projects.restraint.cohort.utils.functions import (
    explode_admissions,
    preprocess_readmissions,
    select_outcomes,
)
from psycop.projects.restraint.cohort.utils.loaders import (
    load_admissions_discharge_timestamps,
    load_coercion_timestamps,
)


class RestraintCohortDefiner(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle:
        unfiltered_prediction_times = pl.LazyFrame(
            pl.from_pandas(load_admissions_discharge_timestamps())
        )

        filtered_prediction_times = filter_prediction_times(
            prediction_times=unfiltered_prediction_times,
            filtering_steps=[RestraintAdmissionTypeFilter(), RestraintMinAgeFilter()],
            entity_id_col_name="dw_ek_borger",
        ).prediction_times.select(  # type: ignore
            pl.col(["dw_ek_borger", "datotid_start", "datotid_slut", "shakkode_ansvarlig"])
        )

        filtered_prediction_times = preprocess_readmissions(df=filtered_prediction_times)

        filtered_prediction_times = filter_prediction_times(
            prediction_times=filtered_prediction_times,
            filtering_steps=[RestraintMinDateFilter()],
            entity_id_col_name="dw_ek_borger",
        ).prediction_times

        unfiltered_coercion_timestamps = pl.LazyFrame(pl.from_pandas(load_coercion_timestamps()))

        filtered_coercion_timestamps = filter_prediction_times(
            prediction_times=unfiltered_coercion_timestamps,
            filtering_steps=[RestraintCoercionTypeFilter()],
            entity_id_col_name="dw_ek_borger",
        ).prediction_times.select(  # type: ignore
            pl.col(["dw_ek_borger", "datotid_start_sei", "typetekst_sei", "behandlingsomraade"])
        )

        unfiltered_cohort = filtered_prediction_times.join(  # type: ignore
            filtered_coercion_timestamps, how="left", on="dw_ek_borger"
        )

        excluded_cohort = filter_prediction_times(
            prediction_times=unfiltered_cohort,
            filtering_steps=[RestraintWashoutFilter()],
            entity_id_col_name="dw_ek_borger",
        ).prediction_times

        excluded_cohort = excluded_cohort.unique(keep="first")  # type: ignore

        filtered_cohort = unfiltered_cohort.join(
            excluded_cohort, how="anti", on=["dw_ek_borger", "datotid_start"]
        )

        cohort_with_coercion = filter_prediction_times(
            prediction_times=filtered_cohort,
            filtering_steps=[RestraintWithinAdmissionsFilter(), RestraintTreatmentUnitFilter()],
            entity_id_col_name="dw_ek_borger",
        ).prediction_times

        cohort_with_outcomes = select_outcomes(cohort_with_coercion)  # type: ignore

        deduplicated_cohort = filtered_cohort.unique(
            subset=["dw_ek_borger", "datotid_start", "datotid_slut"], keep="first"
        ).select(["dw_ek_borger", "datotid_start", "datotid_slut", "shakkode_ansvarlig"])

        filtered_cohort = deduplicated_cohort.join(
            cohort_with_outcomes, how="left", on=["dw_ek_borger", "datotid_start", "datotid_slut"]
        )

        filtered_cohort = filter_prediction_times(
            prediction_times=filtered_cohort,
            filtering_steps=[RestraintAdmissionFilter(), RestraintShakCodeFilter()],
            entity_id_col_name="dw_ek_borger",
        ).prediction_times

        forced_admissions = filter_prediction_times(
            prediction_times=unfiltered_coercion_timestamps,
            filtering_steps=[RestraintForcedAdmissionFilter()],
            entity_id_col_name="dw_ek_borger",
        ).prediction_times.select(  # type: ignore
            ["dw_ek_borger", "datotid_start_sei", "typetekst_sei", "behandlingsomraade"]
        )

        filtered_cohort = filtered_cohort.with_columns(  # type: ignore
            pl.col("datotid_start")
            .dt.strftime("%Y-%m-%d 00:00:00")
            .str.strptime(pl.Datetime("ns"))  # type: ignore
            .alias("dato_start")
        ).select(
            ["dw_ek_borger", "datotid_start", "datotid_slut", "datotid_start_sei", "dato_start"]
        )

        forced_admissions_cohort = filtered_cohort.join(
            forced_admissions,
            how="left",
            left_on=["dw_ek_borger", "dato_start"],
            right_on=["dw_ek_borger", "datotid_start_sei"],
        )

        filtered_forced_admissions_cohort = filter_prediction_times(
            prediction_times=forced_admissions_cohort,
            filtering_steps=[RestraintDoubleAdmissionFilter()],
            entity_id_col_name="dw_ek_borger",
        ).prediction_times.select(  # type: ignore
            ["dw_ek_borger", "datotid_start", "datotid_slut", "datotid_start_sei"]
        )

        exploded_cohort = explode_admissions(filtered_forced_admissions_cohort)

        filtered_exploded_cohort = filter_prediction_times(
            prediction_times=exploded_cohort,
            filtering_steps=[RestraintExcludeDaysFollowingCoercionFilter()],
            entity_id_col_name="dw_ek_borger",
        ).prediction_times

        return filter_prediction_times(
            prediction_times=filtered_exploded_cohort,  # type: ignore
            filtering_steps=[RestraintExcludeFirstDayFilter()],
            entity_id_col_name="dw_ek_borger",
        )

    @staticmethod
    def get_outcome_timestamps() -> OutcomeTimestampFrame:
        unfiltered_prediction_times = pl.LazyFrame(
            pl.from_pandas(load_admissions_discharge_timestamps())
        )

        filtered_prediction_times = filter_prediction_times(
            prediction_times=unfiltered_prediction_times,
            filtering_steps=[RestraintAdmissionTypeFilter(), RestraintMinAgeFilter()],
            entity_id_col_name="dw_ek_borger",
        ).prediction_times

        filtered_prediction_times = preprocess_readmissions(df=filtered_prediction_times).select(  # type: ignore
            ["dw_ek_borger", "datotid_start", "datotid_slut"]
        )

        unfiltered_coercion_timestamps = pl.LazyFrame(pl.DataFrame(load_coercion_timestamps()))

        filtered_coercion_timestamps = filter_prediction_times(
            prediction_times=unfiltered_coercion_timestamps,
            filtering_steps=[RestraintCoercionTypeFilter(), RestraintTreatmentUnitFilter()],
            entity_id_col_name="dw_ek_borger",
        ).prediction_times.select(["dw_ek_borger", "datotid_start_sei", "typetekst_sei"])  # type: ignore

        unfiltered_cohort = filtered_prediction_times.join(
            filtered_coercion_timestamps, how="left", on="dw_ek_borger"
        )

        filtered_cohort = filter_prediction_times(
            prediction_times=unfiltered_cohort,
            filtering_steps=[RestraintWithinAdmissionsFilter()],
            entity_id_col_name="dw_ek_borger",
        ).prediction_times

        outcome_df = select_outcomes(filtered_cohort)  # type: ignore

        # keep adm_id
        return OutcomeTimestampFrame(outcome_df.collect())


if __name__ == "__main__":
    bundle = RestraintCohortDefiner.get_filtered_prediction_times_bundle()

    outcome_timestamps = RestraintCohortDefiner.get_outcome_timestamps()
