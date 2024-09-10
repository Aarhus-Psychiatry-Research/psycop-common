import datetime
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from psycop.common.global_utils.cache import shared_cache
from psycop.common.global_utils.mlflow.mlflow_data_extraction import (
    MlflowClientWrapper,
    PsycopMlflowRun,
)
from psycop.common.model_evaluation.binary.global_performance.roc_auc import bootstrap_roc


@dataclass
class TemporalRunPerformance:
    run: PsycopMlflowRun
    performance: float
    lower_ci: float
    upper_ci: float
    start_date: datetime.datetime
    end_date: datetime.datetime


@shared_cache().cache
def run_perf_from_run(run: PsycopMlflowRun, n_bootstraps: int) -> TemporalRunPerformance:
    eval_frame = run.eval_frame()

    y = eval_frame.frame.to_pandas()["y"]
    y_hat_probs = eval_frame.frame.to_pandas()["y_hat_prob"]
    aucs_bootstrapped, _, _ = bootstrap_roc(n_bootstraps=n_bootstraps, y=y, y_hat_probs=y_hat_probs)

    auc_mean = float(np.mean(aucs_bootstrapped))
    auc_se = np.std(aucs_bootstrapped) / np.sqrt(n_bootstraps)
    auc_ci = [auc_mean - 1.96 * auc_se, auc_mean + 1.96 * auc_se]

    # Get dates
    config = run.get_config()
    date_parser = lambda d: datetime.datetime.strptime(d, "%Y-%m-%d")  # noqa: E731
    start_date = date_parser(
        config.retrieve(
            "trainer.validation_preprocessing_pipeline.*.date_filter_start.threshold_date"
        )
    )
    end_date = date_parser(
        config.retrieve(
            "trainer.validation_preprocessing_pipeline.*.date_filter_end.threshold_date"
        )
    )

    return TemporalRunPerformance(
        run=run,
        performance=auc_mean,
        lower_ci=auc_ci[0],
        upper_ci=auc_ci[1],
        start_date=start_date,
        end_date=end_date,
    )


@shared_cache().cache
def t2d_extended_temporal_stability_model(
    runs: Sequence[PsycopMlflowRun],
) -> Sequence[TemporalRunPerformance]:
    """Temporal stability model for T2D Extended."""
    return [run_perf_from_run(run, n_bootstraps=5) for run in runs]


if __name__ == "__main__":
    client_wrapper = MlflowClientWrapper()
    runs = [
        client_wrapper.get_run(
            experiment_name="t2d_extended", run_name=f"2018-01-01_20{y}-01-01_20{y+1}-01-01"
        )
        for y in range(18, 23)
    ]
    performances = t2d_extended_temporal_stability_model(runs)
