from typing import NewType

import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.common.global_utils.mlflow.mlflow_data_extraction import EvalDF
from psycop.common.model_evaluation.binary.time.timedelta_data import get_auroc_by_timedelta_df
from psycop.projects.cvd.model_evaluation.single_run.sensitivity_by_time_to_event.model import (
    parse_dw_ek_borger_from_uuid,
    parse_timestamp_from_uuid,
)

TimeFromFirstVisitDF = NewType("TimeFromFirstVisitDF", pl.DataFrame)


@shared_cache.cache()
def auroc_by_time_from_first_visit_model(
    eval_df: EvalDF, all_visits_df: pl.DataFrame
) -> TimeFromFirstVisitDF:
    eval_dataset = parse_dw_ek_borger_from_uuid(parse_timestamp_from_uuid(eval_df.frame))

    first_visit = (
        all_visits_df.sort("timestamp", descending=False)
        .groupby("dw_ek_borger")
        .head(1)
        .rename({"timestamp": "first_visit_timestamp"})
    )

    joined_df = eval_dataset.join(
        first_visit.select(["first_visit_timestamp", "dw_ek_borger"]), on="dw_ek_borger", how="left"
    ).to_pandas()

    result_df = get_auroc_by_timedelta_df(
        y=joined_df["y"],
        y_hat_probs=joined_df["y_hat_prob"],
        time_one=joined_df["first_visit_timestamp"],
        time_two=joined_df["timestamp"],
        direction="t2-t1",
        bin_unit="M",
        bins=range(0, 60, 6),
    )

    return TimeFromFirstVisitDF(pl.from_pandas(result_df))
