from collections.abc import Sequence
from typing import NewType

import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.projects.ect.model_evaluation.single_run.auroc_by.auroc_by_model import auroc_by_model
from psycop.projects.ect.model_evaluation.uuid_parsers import (
    parse_dw_ek_borger_from_uuid,
    parse_timestamp_from_uuid,
)

AUROCByAgeDF = NewType("AUROCByAgeDF", pl.DataFrame)


def add_age(df: pl.DataFrame, birthdays: pl.DataFrame, age_col_name: str = "age") -> pl.DataFrame:
    df = df.join(birthdays, on="dw_ek_borger", how="left")
    df = df.with_columns(
        ((pl.col("timestamp") - pl.col("date_of_birth")).dt.days()).alias(age_col_name)
    )
    df = df.with_columns((pl.col(age_col_name) / 365.25).alias(age_col_name))

    return df


@shared_cache().cache()
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
