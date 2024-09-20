from typing import NewType

import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.common.global_utils.mlflow.mlflow_data_extraction import EvalFrame
from psycop.common.model_evaluation.binary.time.absolute_data import (
    create_roc_auc_by_absolute_time_df,
)
from psycop.projects.ect.model_evaluation.uuid_parsers import parse_timestamp_from_uuid

AUROCByQuarterDF = NewType("AUROCByQuarterDF", pl.DataFrame)


@shared_cache().cache()
def auroc_by_quarter_model(eval_frame: EvalFrame) -> AUROCByQuarterDF:
    eval_dataset = parse_timestamp_from_uuid(eval_frame.frame).to_pandas()

    df = create_roc_auc_by_absolute_time_df(
        labels=eval_dataset["y"],
        y_hat_probs=eval_dataset["y_hat_prob"],
        timestamps=eval_dataset["timestamp"],
        bin_period="Y",
    )

    return AUROCByQuarterDF(pl.from_pandas(df))
