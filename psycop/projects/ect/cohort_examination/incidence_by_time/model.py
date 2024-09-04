from typing import NewType

import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.projects.ect.cohort_examination.label_by_outcome_type import label_by_outcome_type
from psycop.projects.ect.feature_generation.cohort_definition.eligible_prediction_times.single_filters import (
    ECTWashoutMove,
)
from psycop.projects.ect.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_ect_indicator,
)

IncidenceByTimeModel = NewType("IncidenceByTimeModel", pl.DataFrame)


@shared_cache().cache()
def incidence_by_time_model() -> IncidenceByTimeModel:
    df_lab_result = get_first_ect_indicator()

    grouped_by_outcome = (
        label_by_outcome_type(pl.from_pandas(df_lab_result), procedure_col="cause")
    )

    filtered_after_move = ECTWashoutMove().apply(grouped_by_outcome.lazy()).collect()

    return IncidenceByTimeModel(filtered_after_move)


if __name__ == "__main__":

    df = incidence_by_time_model()