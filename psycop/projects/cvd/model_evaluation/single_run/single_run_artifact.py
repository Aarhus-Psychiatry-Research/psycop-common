from dataclasses import dataclass
from typing import Protocol

import plotnine
import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import (
    MlflowClientWrapper,
    PsycopMlflowRun,
)

# Must contain columns "y", "y_hat_prob"


@dataclass(frozen=True)
class RunSelector:
    experiment_name: str
    run_name: str


class SingleRunModel:
    def _get_run(self, run: RunSelector) -> PsycopMlflowRun:
        return MlflowClientWrapper().get_run(
            run_name=run.run_name, experiment_name=run.experiment_name
        )

    def get_eval_df(self, run: RunSelector) -> pl.DataFrame:
        return self._get_run(run).eval_df().frame


def get_eval_df(run: RunSelector) -> pl.DataFrame:
    return SingleRunModel().get_eval_df(run=run)


class SingleRunView(Protocol): ...


class SingleRunPlot(SingleRunView):
    def __call__(self) -> plotnine.ggplot: ...


class SingleRunTable(SingleRunView):
    def __call__(self) -> pl.DataFrame: ...


SingleRunArtifact = SingleRunPlot | SingleRunTable
