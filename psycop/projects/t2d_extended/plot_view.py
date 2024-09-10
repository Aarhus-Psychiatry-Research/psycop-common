import logging
from collections.abc import Sequence
from dataclasses import dataclass

import coloredlogs
import plotnine as pn
import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import SingleRunPlot
from psycop.projects.t2d.paper_outputs.config import THEME
from psycop.projects.t2d_extended.plot_model import (
    TemporalRunPerformance,
    t2d_extended_temporal_stability_model,
)


@dataclass(frozen=True)
class T2DExtendedPlot(SingleRunPlot):
    data: Sequence[TemporalRunPerformance]

    def __call__(self) -> pn.ggplot:
        logging.info(f"Starting {self.__class__.__name__}")
        df = pl.concat(run.to_dataframe() for run in self.data)
        mean_score = df["performance"].mean()

        # Plot AUC ROC curve
        plot = (
            pn.ggplot(df, pn.aes(x="start_date", y="performance", ymin="lower_ci", ymax="upper_ci"))
            + pn.geom_point(size=2)
            + pn.stat_smooth(method="lm")
            + pn.labs(y="AUROC", x="Start date")
            + pn.ylim([0.7, 1.0])
            + THEME
            + pn.scale_x_datetime(date_labels="%Y", date_breaks="1 year")
            # + pn.geom_hline(pn.aes(yintercept=mean_score), linetype="dashed")
        )

        plot.save("test.png", width=4, height=2.5, dpi=600)
        return plot


if __name__ == "__main__":
    coloredlogs.install(level="INFO", fmt="%(asctime)s [%(levelname)s] %(message)s")

    client_wrapper = MlflowClientWrapper()
    runs = [
        client_wrapper.get_run(
            experiment_name="T2D-extended, temporal validation",
            run_name=f"2018-01-01_20{y}-01-01_20{y}-12-31",
        )
        for y in range(18, 23)
    ]
    performances = t2d_extended_temporal_stability_model(runs, n_bootstraps=100)

    plot = T2DExtendedPlot(performances)()
