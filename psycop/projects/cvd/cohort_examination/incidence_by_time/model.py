from typing import NewType

import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.projects.cvd.cohort_examination.label_by_outcome_type import label_by_outcome_type
from psycop.projects.cvd.feature_generation.cohort_definition.eligible_prediction_times.single_filters import (
    CVDWashoutMove,
)
from psycop.projects.cvd.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_cvd_indicator,
)

IncidenceByTimeModel = NewType("IncidenceByTimeModel", pl.DataFrame)


@shared_cache().cache()
def incidence_by_time_model() -> IncidenceByTimeModel:
    df_lab_result = get_first_cvd_indicator()

    grouped_by_outcome = (
        label_by_outcome_type(pl.from_pandas(df_lab_result), procedure_col="cause")
        .with_columns(
            pl.when(pl.col("outcome_type").str.contains("artery"))
            .then(pl.lit("PAD"))
            .otherwise("outcome_type")
            .alias("outcome_type")
        )
        .filter(pl.col("outcome_type").is_null().not_())
    )

    filtered_after_move = CVDWashoutMove().apply(grouped_by_outcome.lazy()).collect()

    return IncidenceByTimeModel(filtered_after_move)
