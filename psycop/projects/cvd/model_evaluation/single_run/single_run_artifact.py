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
        return MlflowClientWrapper().get_run(run.run_name, run.experiment_name)

    def _get_eval_df(self, run: RunSelector) -> pl.DataFrame:
        return self._get_run(run).eval_df()


class SingleRunView(Protocol): ...


class SingleRunPlot(SingleRunView):
    def __call__(self) -> plotnine.ggplot: ...


class SingleRunTable(SingleRunView):
    def __call__(self) -> pl.DataFrame: ...


SingleRunArtifact = SingleRunPlot | SingleRunTable
