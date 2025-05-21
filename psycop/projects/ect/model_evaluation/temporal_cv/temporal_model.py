import datetime
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from psycop.common.global_utils.cache import shared_cache
from psycop.common.global_utils.mlflow.mlflow_data_extraction import PsycopMlflowRun
from psycop.common.model_evaluation.binary.global_performance.roc_auc import bootstrap_roc


@dataclass
class TemporalRunPerformance:
    run: PsycopMlflowRun
    performance: float
    train_end_date: datetime.datetime
    start_date: datetime.datetime
    end_date: datetime.datetime

    def to_dataframe(self) -> pl.DataFrame:
        # Convert estimates to a list
        # Create a dictionary with the data
        data = {
            "performance": [self.performance],
            "start_date": [self.start_date],
            "end_date": [self.end_date],
            "train_end_date": [self.train_end_date],
        }

        # Create the DataFrame
        df = pl.DataFrame(data)

        return df


@shared_cache().cache
def run_perf_from_run(run: PsycopMlflowRun) -> TemporalRunPerformance | None:
    
    try:
        auc = run._data.metrics['BinaryAUROC']  # type: ignore
    except KeyError:
        return None  

    # Get dates
    config = run.get_config()
    date_parser = lambda d: datetime.datetime.strptime(d, "%Y-%m-%d")  # noqa: E731
    train_end_date = date_parser(
        config.retrieve("trainer.training_preprocessing_pipeline.*.date_filter.threshold_date")
    )
    start_eval_date = date_parser(
        config.retrieve(
            "trainer.validation_preprocessing_pipeline.*.date_filter_start.threshold_date"
        )
    )
    end_eval_date = date_parser(
        config.retrieve(
            "trainer.validation_preprocessing_pipeline.*.date_filter_end.threshold_date"
        )
    )

    return TemporalRunPerformance(
        run=run,
        performance=auc,
        train_end_date=train_end_date,
        start_date=start_eval_date,
        end_date=end_eval_date,
    )


def temporal_stability(
    runs: Sequence[PsycopMlflowRun]
) -> Sequence[TemporalRunPerformance]:
    """Temporal stability model for T2D Extended."""
    return [auc for run in runs if (auc := run_perf_from_run(run)) is not None] 
