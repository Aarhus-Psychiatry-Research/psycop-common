from typing import NewType

import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.common.global_utils.mlflow.mlflow_data_extraction import EvalDF
from psycop.common.model_evaluation.binary.time.periodic_data import roc_auc_by_periodic_time_df
from psycop.projects.cvd.model_evaluation.single_run.sensitivity_by_time_to_event.model import (
    parse_timestamp_from_uuid,
)

AUROCByDayOfWeekDF = NewType("AUROCByDayOfWeekDF", pl.DataFrame)


@shared_cache.cache()
def auroc_by_day_of_week_model(eval_df: EvalDF) -> AUROCByDayOfWeekDF:
    eval_dataset = parse_timestamp_from_uuid(eval_df).to_pandas()

    df = roc_auc_by_periodic_time_df(
        labels=eval_dataset["y"],
        y_hat_probs=eval_dataset["y_hat_prob"],
        timestamps=eval_dataset["timestamp"],
        bin_period="D",
    )

    return AUROCByDayOfWeekDF(pl.from_pandas(df))
