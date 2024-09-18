import datetime
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import coloredlogs
import plotnine as pn
import polars as pl
import statsmodels.api as sm

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import SingleRunPlot
from psycop.projects.t2d.paper_outputs.config import THEME
from psycop.projects.t2d_extended.temporal_model import (
    TemporalRunPerformance,
    t2d_extended_temporal_stability_model,
)


@dataclass(frozen=True)
class T2DExtendedPlot(SingleRunPlot):
    data: Sequence[TemporalRunPerformance]

    def __call__(self) -> pn.ggplot:
        logging.info(f"Starting {self.__class__.__name__}")
        df = pl.concat(run.to_dataframe() for run in self.data)

        estimates = df.explode("estimates").with_columns(
            (
                (pl.col("start_date").cast(pl.Date) - pl.col("start_date").min()).dt.total_days()
                / 365.25
            ).alias("years_since_start")
        )

        def format_datetime(x: Sequence[datetime.datetime]) -> Sequence[str]:
            return [f"{d.year}" for d in x]

        # Convert to numpy arrays for scikit-learn
        X = estimates.to_pandas()["years_since_start"].to_numpy().reshape(-1, 1)
        y = estimates.to_pandas()["estimates"].to_numpy()

        # Use statsmodels to get the standard error of the slope
        X_with_const = sm.add_constant(X)
        sm_model = sm.GLM(y, X_with_const).fit()
        slope = sm_model.params[1]  # type: ignore
        slope_se = sm_model.bse[1]  # type: ignore

        # Plot AUC ROC curve
        plot = (
            pn.ggplot(df, pn.aes(x="start_date", y="performance", ymin="lower_ci", ymax="upper_ci"))
            + pn.lims(y=(0.7, 1.0))
            + pn.geom_point(size=2)
            + pn.stat_smooth(
                data=estimates, mapping=pn.aes(x="start_date", y="estimates"), se=True, size=0.5
            )
            + pn.labs(y="AUROC", x="")
            + THEME
            + pn.scale_x_datetime(labels=format_datetime, date_breaks="1 year")
            + pn.theme(
                panel_grid_major=pn.element_line(color="lightgrey", size=0.25, linetype="dotted"),
                axis_text_x=pn.element_text(size=10),
            )
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
    performances = t2d_extended_temporal_stability_model(runs, n_bootstraps=500)

    plot = T2DExtendedPlot(performances)()
