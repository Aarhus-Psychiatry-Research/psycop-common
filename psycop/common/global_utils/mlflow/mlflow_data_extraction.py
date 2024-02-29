from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import mlflow
import polars as pl
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from psycop.common.types.validated_frame import ValidatedFrame


@dataclass(frozen=True)
class MlflowAllMetricsFrame(ValidatedFrame[pl.DataFrame]):
    frame: pl.DataFrame

    run_name_col_name: str = "run_name"
    metric_col_name: str = "metric"
    value_col_name: str = "value"

    allow_extra_columns = False


@dataclass(frozen=True)
class MlflowTimeVaryingMetricsFrame(ValidatedFrame[pl.DataFrame]):
    frame: pl.DataFrame

    run_name_col_name: str = "run_name"
    metric_col_name: str = "metric"
    value_col_name: str = "value"
    unix_timestamp_col_name: str = "timestamp"
    step_col_name: str = "step"

    allow_extra_columns = False


class MlflowMetricExtractor:
    def __init__(self) -> None:
        tracking_uri = "http://exrhel0371.it.rm.dk:5050"

        self.client = MlflowClient(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)

    def get_all_metrics_for_experiment(self, experiment_name: str) -> MlflowAllMetricsFrame:
        """Get the final value of all logged metrics for each run in an experiment.
        Returns a long df with cols 'run_name', 'metric', and 'value'."""
        runs = self._get_mlflow_runs_by_experiment(experiment_name=experiment_name)
        metrics_df = pl.concat([self._get_all_metrics_for_run(run) for run in runs]).melt(
            id_vars="run_name", variable_name="metric"
        )

        return MlflowAllMetricsFrame(frame=metrics_df, allow_extra_columns=False)

    def get_metrics_for_experiment(
        self, experiment_name: str, metrics: Iterable[str]
    ) -> MlflowTimeVaryingMetricsFrame:
        """Get all values of specified metrics for each run in an experiment. This
        includes values that get updated such as loss. Returns a long df with
        cols 'run_name', 'metric', 'value', 'timestamp', 'step'."""
        runs = self._get_mlflow_runs_by_experiment(experiment_name=experiment_name)
        metrics_df = pl.concat(
            [self._get_metrics_for_run(run=run, metrics=metrics) for run in runs]
        ).rename({"key": "metric"})

        return MlflowTimeVaryingMetricsFrame(frame=metrics_df, allow_extra_columns=False)

    def get_best_run_from_experiments(self, experiment_names: Iterable[str], metric: str) -> Run:
        """Get the best run from one or more experiments based on some metric,
        e.g. 'all_oof_BinaryAUROC'"""
        experiment_ids = [
            self._get_mlflow_experiment_id_from_experiment_name(experiment_name)
            for experiment_name in experiment_names
        ]

        best_run = self.client.search_runs(
            experiment_ids=experiment_ids, max_results=1, order_by=[f"metrics.{metric} DESC"]
        )[0]
        return best_run

    def download_config_from_best_run_from_experiments(
        self, experiment_names: Iterable[str], metric: str, save_location: str | None = None
    ) -> Path:
        """Download the config from the best from a list of experiments. Returns the path to the config.
        If save_location is None, will save to temporary directory"""
        best_run = self.get_best_run_from_experiments(
            experiment_names=experiment_names, metric=metric
        )
        return self.download_artifact_from_run(run=best_run, artifact_name="config.cfg", save_location=save_location)

    def download_artifact_from_run(
        self, run: Run, artifact_name: str, save_location: str | None = None
    ) -> Path:
        """Download an artifact from a run. Eeturns the path to the downloaded artifact. 
        If save_location is None, will save to temporary directory"""
        return Path(
            self.client.download_artifacts(
                run_id=run.info.run_id, path=artifact_name, dst_path=save_location
            )
        )

    def _get_metrics_for_run(self, run: Run, metrics: Iterable[str]) -> pl.DataFrame:
        metrics_df = [
            self._get_metric_for_run(run_id=run.info.run_id, metric=metric) for metric in metrics
        ]
        return pl.concat(metrics_df).with_columns(pl.lit(run.info.run_name).alias("run_name"))

    def _get_metric_for_run(self, run_id: str, metric: str) -> pl.DataFrame:
        return pl.DataFrame(
            [
                dict(metric_object)
                for metric_object in self.client.get_metric_history(run_id=run_id, key=metric)
            ]
        )

    def _get_all_metrics_for_run(self, run: Run) -> pl.DataFrame:
        return pl.DataFrame(run.data.metrics).with_columns(
            pl.lit(run.info.run_name).alias("run_name")
        )

    def _get_mlflow_experiment_id_from_experiment_name(self, experiment_name: str) -> str:
        experiment = self.client.get_experiment_by_name(name=experiment_name)
        if experiment is None:
            raise ValueError(f"{experiment_name} does not exist on MlFlow.")
        return experiment.experiment_id

    def _get_mlflow_runs_by_experiment(self, experiment_name: str) -> Iterable[Run]:
        experiment_id = self._get_mlflow_experiment_id_from_experiment_name(
            experiment_name=experiment_name
        )
        return self.client.search_runs(experiment_ids=[experiment_id])


if __name__ == "__main__":
    df = MlflowMetricExtractor().get_all_metrics_for_experiment("text_exp")
    df = MlflowMetricExtractor().get_metrics_for_experiment(
        "text_exp", ["all_oof_BinaryAUROC", "within_fold_0_BinaryAUROC"]
    )
    best_run = MlflowMetricExtractor().get_best_run_from_experiments(
        experiment_names=["text_exp"], metric="all_oof_BinaryAUROC"
    )
