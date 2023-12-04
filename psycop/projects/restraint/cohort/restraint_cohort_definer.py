import polars as pl

from psycop.common.cohort_definition import (
    CohortDefiner,
    FilteredPredictionTimeBundle,
    filter_prediction_times,
)
from psycop.projects.restraint.cohort.utils.filters import (
    RestraintAdmissionFilter,
    RestraintAdmissionTypeFilter,
    RestraintCoercionTypeFilter,
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


def load_prediction_times() -> pl.DataFrame:
    unfiltered_prediction_times = pl.from_pandas(load_admissions_discharge_timestamps())

    filtered_prediction_times = filter_prediction_times(
        prediction_times=unfiltered_prediction_times,
        filtering_steps=[
            RestraintAdmissionTypeFilter(),
            RestraintMinDateFilter(),
            RestraintMinAgeFilter(),
        ],
        entity_id_col_name="dw_ek_borger",
    ).prediction_times.select(
        pl.col(["dw_ek_borger", "datotid_start", "datotid_slut", "shakkode_ansvarlig"])
    )

    filtered_prediction_times = preprocess_readmissions(df=filtered_prediction_times)

    unfiltered_coercion_timestamps = pl.from_pandas(load_coercion_timestamps())

    filtered_coercion_timestamps = filter_prediction_times(
        prediction_times=unfiltered_coercion_timestamps,
        filtering_steps=[RestraintCoercionTypeFilter()],
        entity_id_col_name="dw_ek_borger",
    ).prediction_times.select(
        pl.col(
            ["dw_ek_borger", "datotid_start_sei", "typetekst_sei", "behandlingsomraade"],
        ),
    )

    unfiltered_cohort = filtered_prediction_times.join(
        filtered_coercion_timestamps,
        how="left",
        on="dw_ek_borger",
    )

    excluded_cohort = filter_prediction_times(
        prediction_times=unfiltered_cohort,
        filtering_steps=[RestraintWashoutFilter()],
        entity_id_col_name="dw_ek_borger",
    ).prediction_times

    excluded_cohort = excluded_cohort.unique(keep="first")

    unfiltered_cohort = unfiltered_cohort.join(
        excluded_cohort,
        how="anti",
        on=["dw_ek_borger", "datotid_start"],
    )

    deduplicated_cohort = unfiltered_cohort.unique(
        subset=["dw_ek_borger", "datotid_start", "datotid_slut"],
        keep="first",
    )

    filtered_cohort = filter_prediction_times(
        prediction_times=deduplicated_cohort,
        filtering_steps=[RestraintAdmissionFilter()],
        entity_id_col_name="dw_ek_borger",
    ).prediction_times

    exploded_cohort = explode_admissions(filtered_cohort)

    # unfiltered_prediction_times = pl.from_pandas(
    #     load_admissions_discharge_timestamps()
    # ).select(pl.col(["dw_ek_borger", "datotid_start", "shakkode_ansvarlig"]))

    return exploded_cohort.join(
        unfiltered_prediction_times,
        how="left",
        on=["dw_ek_borger", "datotid_start"],
    )


class RestraintCohortDefiner(CohortDefiner):
    @staticmethod
    def get_filtered_prediction_times_bundle() -> FilteredPredictionTimeBundle:
        unfiltered_cohort = load_prediction_times()

        return filter_prediction_times(
            prediction_times=unfiltered_cohort,
            filtering_steps=[RestraintShakCodeFilter(), RestraintTreatmentUnitFilter()],
            entity_id_col_name="dw_ek_borger",
        )

    @staticmethod
    def get_outcome_timestamps() -> pl.DataFrame:
        unfiltered_cohort = load_prediction_times()

        filtered_cohort = filter_prediction_times(
            prediction_times=unfiltered_cohort,
            filtering_steps=[RestraintWithinAdmissionsFilter()],
            entity_id_col_name="dw_ek_borger",
        )

        outcome_df = select_outcomes(filtered_cohort.prediction_times)

        # keep adm_id
        return outcome_df


if __name__ == "__main__":
    bundle = RestraintCohortDefiner.get_filtered_prediction_times_bundle()

    outcome_timestamps = RestraintCohortDefiner.get_outcome_timestamps()

    print("i")
