import datetime
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import coloredlogs
import plotnine as pn
import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import SingleRunPlot
from psycop.projects.forced_admission_inpatient.model_eval.temporal_model import (
    TemporalRunPerformance,
    temporal_stability,
)
from psycop.projects.t2d.paper_outputs.config import THEME


@dataclass(frozen=True)
class TemporalStabilityPlot(SingleRunPlot):
    data: Sequence[TemporalRunPerformance]

    def __call__(self) -> pn.ggplot:
        logging.info(f"Starting {self.__class__.__name__}")
        df = pl.concat(run.to_dataframe() for run in self.data)

        df = df.with_columns(
            pl.col("train_end_date").dt.year().cast(pl.Utf8).alias("train_end_year"),
            (pl.col("start_date") - (pl.col("train_end_date"))).alias("since_train_end"),
        )

        def format_datetime(x: Sequence[datetime.datetime]) -> Sequence[str]:  # type: ignore
            return [f"{d.year-2000}-{d.year+1-2000}" for d in x]

        plot = (
            pn.ggplot(
                df,
                pn.aes(
                    x="since_train_end",
                    y="performance",
                    ymin="lower_ci",
                    ymax="upper_ci",
                    color="train_end_year",
                ),
            )
            + pn.scale_color_ordinal()
            + pn.geom_line()
            + pn.geom_point(size=2)
            + pn.labs(y="AUROC", x="")
            + THEME
            # + pn.scale_x_datetime(labels=format_datetime, date_breaks="1 year")  # noqa: ERA001
            + pn.theme(
                panel_grid_major=pn.element_line(color="lightgrey", size=0.25, linetype="dotted"),
                axis_text_x=pn.element_text(size=10),
            )
        )

        plot.save("test.png", width=4, height=2.5, dpi=600)
        return plot


if __name__ == "__main__":
    coloredlogs.install(level="INFO", fmt="%(asctime)s [%(levelname)s] %(message)s")

    runs = MlflowClientWrapper().get_runs_from_experiment("Inpatient v2, temporal validation")

    performances = temporal_stability(runs, n_bootstraps=50)

    plot = TemporalStabilityPlot(performances)()
