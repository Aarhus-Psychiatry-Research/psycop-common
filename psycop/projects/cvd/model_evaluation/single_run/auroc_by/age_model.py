from typing import NewType

import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.auroc_by_model import auroc_by_model
from psycop.projects.cvd.model_evaluation.single_run.sensitivity_by_time_to_event.model import (
    add_age,
    parse_dw_ek_borger_from_uuid,
)

AurocByAgeDF = NewType("AurocByAgeDF", pl.DataFrame)


@shared_cache.cache()
def auroc_by_age_model(eval_df: pl.DataFrame, age_df: pl.DataFrame) -> AurocByAgeDF:
    eval_dataset = (add_age(parse_dw_ek_borger_from_uuid(eval_df), age_df)).to_pandas()

    df = auroc_by_model(
        input_values=eval_dataset["age"],
        y=eval_dataset["y"],
        y_hat_probs=eval_dataset["y_hat_prob"],
        input_name="age",
    )

    return AurocByAgeDF(pl.from_pandas(df))
