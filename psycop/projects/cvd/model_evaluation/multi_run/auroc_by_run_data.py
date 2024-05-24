import logging
from collections.abc import Sequence
from typing import NewType

import numpy as np
import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_evaluation.binary.global_performance.roc_auc import bootstrap_roc
from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import RunSelector


def _run_auroc_with_ci(df: pl.DataFrame, n_bootstraps: int = 5) -> pl.DataFrame:
    logging.info(f"Bootstrapping {df['run_name'][0]}")
    _, aucs_bootstrapped, _ = bootstrap_roc(
        n_bootstraps=n_bootstraps,
        y=df["y"],  # type: ignore
        y_hat_probs=df["y_hat_prob"],  # type: ignore
    )

    # Calculate confidence interval for AUC
    auc_mean = np.mean(aucs_bootstrapped)
    auc_se = np.std(aucs_bootstrapped) / np.sqrt(n_bootstraps)
    auc_ci = [auc_mean - 1.96 * auc_se, auc_mean + 1.96 * auc_se]

    return pl.DataFrame(
        {"run_name": df["run_name"][0], "auroc": auc_mean, "lower": auc_ci[0], "upper": auc_ci[1]}
    )


def data(runs: Sequence[RunSelector]) -> pl.DataFrame:
    eval_dfs = [
        MlflowClientWrapper()
        .get_run(r.experiment_name, r.run_name)
        .eval_frame()
        .frame.with_columns(pl.lit(r.run_name).alias("run_name"))
        for r in runs
    ]
    run_performances = [_run_auroc_with_ci(df=df) for df in eval_dfs]
    return pl.concat(run_performances)


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    result = data(
        runs=[
            RunSelector(experiment_name="baseline_v2_cvd", run_name="Layer 1"),
            RunSelector(
                experiment_name="baseline_v2_cvd", run_name="Layer 1 + agg (min, mean, max)"
            ),
            RunSelector(
                experiment_name="baseline_v2_cvd", run_name="Layer 1 + lookbehinds (90, 365, 730)"
            ),
            RunSelector(experiment_name="baseline_v2_cvd", run_name="Layer 2"),
            RunSelector(experiment_name="baseline_v2_cvd", run_name="Layer 3"),
            RunSelector(experiment_name="baseline_v2_cvd", run_name="Layer 4"),
        ]
    )
