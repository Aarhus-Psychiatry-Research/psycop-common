from typing import Iterable

import polars as pl
from mlflow.entities import Run
from mlflow.store.entities import PagedList
from mlflow.tracking import MlflowClient


class MlflowDataExtraction:
    def __init__(self) -> None:
        self.client = MlflowClient(tracking_uri="http://exrhel0371.it.rm.dk:5050")

    def get_all_metrics_for_experiment(self, experiment_name: str) -> pl.DataFrame:
        """Get the final value of all logged metrics for each run in an experiment.
        Returns a long df with cols 'run_name', 'metric', and 'value'."""
        runs = self._get_mlflow_runs_by_experiment(experiment_name=experiment_name)
        metrics_df = [self._get_all_metrics_for_run(run) for run in runs]
        return pl.concat(metrics_df).melt(id_vars="run_name", variable_name="metric")

    def get_metrics_for_experiment(
        self, experiment_name: str, metrics: Iterable[str]
    ) -> pl.DataFrame:
        """Get all values of specified metrics for each run in an experiment. This
        includes values that get updated such as loss. Returns a long df with
        cols 'run_name', 'metric', 'value', 'timestamp', 'step'."""
        runs = self._get_mlflow_runs_by_experiment(experiment_name=experiment_name)
        metrics_df = [self._get_metrics_for_run(run=run, metrics=metrics) for run in runs]
        return pl.concat(metrics_df).rename({"key": "metric"})

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

    def _get_mlflow_runs_by_experiment(self, experiment_name: str) -> PagedList[Run]:
        experiment_id = self._get_mlflow_experiment_id_from_experiment_name(
            experiment_name=experiment_name
        )
        return self.client.search_runs(experiment_ids=[experiment_id])


if __name__ == "__main__":
    df = MlflowDataExtraction().get_all_metrics_for_experiment("text_exp")
    df = MlflowDataExtraction().get_metrics_for_experiment(
        "text_exp", ["all_oof_BinaryAUROC", "within_fold_0_BinaryAUROC"]
    )
    
