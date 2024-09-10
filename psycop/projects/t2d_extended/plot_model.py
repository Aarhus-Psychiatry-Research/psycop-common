import datetime
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.common.global_utils.mlflow.mlflow_data_extraction import PsycopMlflowRun
from psycop.common.model_evaluation.binary.global_performance.roc_auc import bootstrap_roc


@dataclass
class TemporalRunPerformance:
    run: PsycopMlflowRun
    performance: float
    lower_ci: float
    upper_ci: float
    estimates: np.ndarray[Any, Any]
    start_date: datetime.datetime
    end_date: datetime.datetime

    def to_dataframe(self) -> pl.DataFrame:
        # Convert estimates to a list
        estimates_list = self.estimates.tolist()

        flattened = list(itertools.chain.from_iterable(estimates_list))

        # Create a dictionary with the data
        data = {
            "performance": [self.performance],
            "lower_ci": [self.lower_ci],
            "upper_ci": [self.upper_ci],
            "start_date": [self.start_date],
            "end_date": [self.end_date],
        }

        # Create the DataFrame
        df = pl.DataFrame(data)

        return df


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
        estimates=aucs_bootstrapped,
    )


def t2d_extended_temporal_stability_model(
    runs: Sequence[PsycopMlflowRun], n_bootstraps: int
) -> Sequence[TemporalRunPerformance]:
    """Temporal stability model for T2D Extended."""
    return [run_perf_from_run(run, n_bootstraps=n_bootstraps) for run in runs]
