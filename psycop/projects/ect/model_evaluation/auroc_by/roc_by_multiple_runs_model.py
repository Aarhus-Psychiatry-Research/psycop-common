from typing import NewType

import polars as pl
from sklearn.metrics import roc_auc_score, roc_curve

from psycop.common.global_utils.cache import shared_cache
from psycop.common.global_utils.mlflow.mlflow_data_extraction import EvalFrame

ROCByGroupDF = NewType("ROCByGroupDF", pl.DataFrame)
ExperimentWithNames = NewType("ExperimentWithNames", dict[str, EvalFrame])


@shared_cache().cache()
def _run_auroc_df(df: pl.DataFrame) -> pl.DataFrame:
    fpr, tpr, _ = roc_curve(y_true=df["y"].to_numpy(), y_score=df["y_hat_prob"].to_numpy())
    auroc_df = pl.DataFrame({"fpr": fpr, "tpr": tpr}).with_columns(
        pl.lit(df["run_name"][0]).alias("run_name"),
        pl.lit(roc_auc_score(y_true=df["y"], y_score=df["y_hat_prob"])).alias("AUROC"),
    )
    return auroc_df


def _add_run_name(eval_frame: EvalFrame, name: str) -> pl.DataFrame:
    return eval_frame.frame.with_columns(pl.lit(name).alias("run_name"))


@shared_cache().cache()
def group_auroc_model(runs: ExperimentWithNames) -> ROCByGroupDF:
    eval_dfs = [_add_run_name(run, name) for name, run in runs.items()]
    run_performances = [_run_auroc_df(df=df) for df in eval_dfs]
    return ROCByGroupDF(pl.concat(run_performances))
