import logging
from collections.abc import Sequence
from dataclasses import dataclass

import plotnine as pn
import polars as pl

from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import SingleRunPlot
from psycop.projects.t2d.paper_outputs.config import THEME
from psycop.projects.t2d_extended.plot_model import TemporalRunPerformance


@dataclass(frozen=True)
class T2DExtendedPlot(SingleRunPlot):
    data: Sequence[TemporalRunPerformance]

    def __call__(self) -> pn.ggplot:
        logging.info(f"Starting {self.__class__.__name__}")
        df = pl.DataFrame(
            {
                "run": [run.run.run_name for run in self.data],  # type: ignore
                "auroc": [run.performance for run in self.data],
                "lower_ci": [run.lower_ci for run in self.data],
                "upper_ci": [run.upper_ci for run in self.data],
                "start_date": [run.start_date for run in self.data],
                "end_date": [run.end_date for run in self.data],
            }
        )

        # Plot AUC ROC curve
        return (
            pn.ggplot(df, pn.aes(x="start_date", y="auroc", ymin="lower", ymax="upper"))
            + pn.geom_point(size=4)
            + pn.geom_errorbar(width=0.3)
            + pn.labs(y="AUROC")
            + THEME
        )
