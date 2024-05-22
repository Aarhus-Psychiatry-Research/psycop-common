from collections.abc import Sequence
from typing import NewType

import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.auroc_by_model import auroc_by_model
from psycop.projects.cvd.model_evaluation.single_run.sensitivity_by_time_to_event.model import (
    add_age,
    parse_dw_ek_borger_from_uuid,
)
from psycop.projects.cvd.model_evaluation.single_run.sensitivity_by_time_to_event.parse_timestamp_from_uuid import (
    parse_timestamp_from_uuid,
)

AUROCByAgeDF = NewType("AUROCByAgeDF", pl.DataFrame)


@shared_cache.cache()
def auroc_by_age_model(
    eval_df: pl.DataFrame, birthdays: pl.DataFrame, bins: Sequence[float]
) -> AUROCByAgeDF:
    eval_dataset = (
        add_age(parse_timestamp_from_uuid(parse_dw_ek_borger_from_uuid(eval_df)), birthdays)
    ).to_pandas()

    df = auroc_by_model(
        input_values=eval_dataset["age"],
        y=eval_dataset["y"],
        y_hat_probs=eval_dataset["y_hat_prob"],
        input_name="age",
        bins=bins,
    )

    return AUROCByAgeDF(pl.from_pandas(df))
