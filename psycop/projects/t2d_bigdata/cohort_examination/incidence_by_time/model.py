from typing import NewType

import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.projects.t2d_bigdata.feature_generation.cohort_definition.eligible_prediction_times.single_filters import (
    T2DWashoutMove,
)
from psycop.projects.t2d_bigdata.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_diabetes_lab_result_above_threshold,
)

IncidenceByTimeModel = NewType("IncidenceByTimeModel", pl.DataFrame)


@shared_cache().cache()
def incidence_by_time_model() -> IncidenceByTimeModel:
    df_lab_result = get_first_diabetes_lab_result_above_threshold()

    # grouped_by_outcome = (  # noqa: ERA001
    #     label_by_outcome_type(pl.from_pandas(df_lab_result), procedure_col="cause") # TODO add type2prodecures dict for T2D (based on what to group by)  # noqa: ERA001
    #     .with_columns(
    #         pl.when(pl.col("outcome_type").str.contains("artery"))  # noqa: ERA001
    #         .then(pl.lit("PAD"))
    #         .otherwise("outcome_type")
    #         .alias("outcome_type")
    #     )  # noqa: ERA001
    #     .filter(pl.col("outcome_type").is_null().not_())
    # )  # noqa: ERA001

    filtered_after_move = T2DWashoutMove().apply(pl.from_pandas(df_lab_result).lazy()).collect()

    return IncidenceByTimeModel(filtered_after_move)
