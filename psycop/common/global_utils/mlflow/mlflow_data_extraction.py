import pathlib
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Callable

import mlflow
import polars as pl
from mlflow.entities import Run
from mlflow.entities.run_data import RunData
from mlflow.entities.run_info import RunInfo
from mlflow.entities.run_inputs import RunInputs
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from psycop.common.model_training_v2.config.config_utils import (
    PsycopConfig,
    resolve_and_fill_config,
)
from psycop.common.types.validated_frame import ValidatedFrame
from psycop.common.types.validator_rules import ColumnExistsRule, ColumnTypeRule, ValidatorRule


@dataclass(frozen=True)
class MlflowAllMetricsFrame(ValidatedFrame[pl.DataFrame]):
    frame: pl.DataFrame

    run_name_col_name: str = "run_name"
    metric_col_name: str = "metric"
    value_col_name: str = "value"

    allow_extra_columns = False


@dataclass(frozen=True)
class EvalFrame(ValidatedFrame[pl.DataFrame]):
    y_col_name: str = "y"
    y_hat_prob_col_name: str = "y_hat_prob"
    pred_time_uuid_col_name: str = "pred_time_uuid"

    frame: pl.DataFrame

    pred_time_uuid_col_rules: Sequence[ValidatorRule] = (
        ColumnExistsRule(),
        ColumnTypeRule(expected_type=pl.Utf8),
    )
    # pred_time_uuid: a string of the form "{citizen id}-%Y-%m-%d-%H-%M-%S", e.g. "98573-2021-01-01-00-00-00"

    y_col_rules: Sequence[ValidatorRule] = (
        ColumnExistsRule(),
        ColumnTypeRule(expected_type=pl.Int64),
    )

    y_hat_prob_col_rules: Sequence[ValidatorRule] = (
        ColumnExistsRule(),
        ColumnTypeRule(expected_type=pl.Float64),
    )


class PsycopMlflowRun(Run):
    def __init__(
        self,
        run_info: RunInfo,
        run_data: RunData,
        run_inputs: RunInputs | None,
        client: MlflowClient,
    ) -> None:
        super().__init__(run_info, run_data, run_inputs)
        self._client = client

    @classmethod
    def from_mlflow_run(
        cls: type["PsycopMlflowRun"], run: Run, client: MlflowClient
    ) -> "PsycopMlflowRun":
        return cls(run_info=run._info, run_data=run._data, run_inputs=run._inputs, client=client)

    @property
    def name(self) -> str:
        return self._info._run_name  # type: ignore

    def get_config(self) -> PsycopConfig:
        cfg_path = self.download_artifact(artifact_name="config.cfg", save_location=None)
        return PsycopConfig().from_disk(cfg_path)

    def sklearn_pipeline(self) -> Pipeline:
        return pickle.load(self.download_artifact("sklearn_pipe.pkl").open("rb"))

    def eval_frame(self) -> EvalFrame:
        eval_df_path = self.download_artifact(artifact_name="eval_df.parquet", save_location=None)
        return EvalFrame(frame=pl.read_parquet(eval_df_path), allow_extra_columns=False)

    def download_artifact(self, artifact_name: str, save_location: str | None = None) -> Path:
        """Download an artifact from a run. Returns the path to the downloaded artifact.
        If save_location is None, will save to temporary directory"""
        return Path(
            self._client.download_artifacts(
                run_id=self.info.run_id, path=artifact_name, dst_path=save_location
            )
        )

    def get_all_metrics(self) -> pl.DataFrame:
        return pl.DataFrame(self.data.metrics).with_columns(
            pl.lit(self.info.run_name).alias("run_name")
        )


class MlflowClientWrapper:
    def __init__(self) -> None:
        tracking_uri = "http://exrhel0371.it.rm.dk:5050"

        self.client = MlflowClient(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)

    def get_all_metrics_for_experiment(self, experiment_name: str) -> MlflowAllMetricsFrame:
        """Get the final value of all logged metrics for each run in an experiment.
        Returns a long df with cols 'run_name', 'metric', and 'value'."""
        runs = self._get_mlflow_runs_by_experiment(experiment_name=experiment_name)
        metrics_df = pl.concat([run.get_all_metrics() for run in runs]).melt(
            id_vars="run_name", variable_name="metric"
        )

        return MlflowAllMetricsFrame(frame=metrics_df, allow_extra_columns=False)

    def get_run(self, experiment_name: str, run_name: str) -> PsycopMlflowRun:
        runs = self._get_mlflow_runs_by_experiment(experiment_name=experiment_name)

        matches = [run for run in runs if run.info.run_name == run_name]
        if not matches:
            raise ValueError(f"No run in experiment {experiment_name} matches {run_name}")
        if len(matches) != 1:
            raise ValueError(f"Multiple runs in experiment {experiment_name} matches {run_name}")

        return matches[0]

    def get_best_run_from_experiment(
        self, experiment_name: str, metric: str, larger_is_better: bool = True
    ) -> PsycopMlflowRun:
        """Get the best run from one or more experiments based on some metric,
        e.g. 'all_oof_BinaryAUROC'"""
        order = "DESC" if larger_is_better else "ASC"

        experiment_id = self._get_mlflow_experiment_id_from_experiment_name(experiment_name)

        best_run = self.client.search_runs(
            experiment_ids=[experiment_id], max_results=1, order_by=[f"metrics.{metric} {order}"]
        )[0]
        return PsycopMlflowRun.from_mlflow_run(run=best_run, client=self.client)

    def _get_mlflow_experiment_id_from_experiment_name(self, experiment_name: str) -> str:
        experiment = self.client.get_experiment_by_name(name=experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment {experiment_name} does not exist on MlFlow.")
        return experiment.experiment_id

    def _get_mlflow_runs_by_experiment(self, experiment_name: str) -> Sequence[PsycopMlflowRun]:
        experiment_id = self._get_mlflow_experiment_id_from_experiment_name(
            experiment_name=experiment_name
        )
        runs = self.client.search_runs(experiment_ids=[experiment_id])
        return [PsycopMlflowRun.from_mlflow_run(run=run, client=self.client) for run in runs]


if __name__ == "__main__":
    df = MlflowClientWrapper().get_all_metrics_for_experiment("text_exp")

    best_config = (
        MlflowClientWrapper()
        .get_best_run_from_experiment(
            experiment_name="scz-bp_3_year_lookahead", metric="all_oof_BinaryAUROC"
        )
        .get_config()
    )


def filled_cfg_from_run(
    run: PsycopMlflowRun, populate_registry_fns: Sequence[Callable[[], None]]
) -> dict[str, Any]:
    cfg = run.get_config()
    for populate_registry_fn in populate_registry_fns:
        populate_registry_fn()

    tmp_cfg = pathlib.Path(mkdtemp()) / "tmp.cfg"
    cfg.to_disk(tmp_cfg)
    filled = resolve_and_fill_config(tmp_cfg, fill_cfg_with_defaults=True)

    return filled
