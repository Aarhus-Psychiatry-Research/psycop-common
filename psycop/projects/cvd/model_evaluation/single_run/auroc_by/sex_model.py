from typing import NewType

import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.auroc_by_model import auroc_by_model
from psycop.projects.cvd.model_evaluation.single_run.sensitivity_by_time_to_event.model import (
    parse_dw_ek_borger_from_uuid,
)

AurocBySexDF = NewType("AurocBySexDF", pl.DataFrame)


@shared_cache.cache()
def auroc_by_sex_model(eval_df: pl.DataFrame, sex_df: pl.DataFrame) -> AurocBySexDF:
    eval_dataset = (
        parse_dw_ek_borger_from_uuid(eval_df)
        .join(sex_df, on="dw_ek_borger", how="left")
        .to_pandas()
    )

    df = auroc_by_model(
        input_values=eval_dataset["sex_female"],
        y=eval_dataset["y"],
        y_hat_probs=eval_dataset["y_hat_prob"],
        input_name="sex",
        bin_continuous_input=False,
    )

    return AurocBySexDF(pl.from_pandas(df))
