import datetime
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import coloredlogs
import plotnine as pn
import polars as pl

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.projects.ect.model_evaluation.single_run_artifact import SingleRunPlot
from psycop.projects.ect.model_evaluation.temporal_cv.temporal_model import (
    TemporalRunPerformance,
    temporal_stability,
)
from psycop.projects.t2d.paper_outputs.config import THEME


@dataclass(frozen=True)
class TemporalStabilityPlot(SingleRunPlot):
    data: Sequence[TemporalRunPerformance]
    relative_time: bool

    def __call__(self) -> pn.ggplot:
        logging.info(f"Starting {self.__class__.__name__}")
        df = pl.concat(run.to_dataframe() for run in self.data)

        df = df.with_columns(
            (pl.col("train_end_date").dt.year() - 1).cast(pl.Utf8).alias("train_end_year"),
            (pl.col("start_date") - (pl.col("train_end_date"))).alias("since_train_end"),
        )

        def format_datetime(x: Sequence[datetime.datetime]) -> Sequence[str]:  # type: ignore
            return [f"{d.year}" for d in x]

        x_variable = "since_train_end" if self.relative_time else "start_date"

        plot = (
            pn.ggplot(
                df, pn.aes(x=x_variable, y="performance", ymin=0.5, ymax=1, color="train_end_year")
            )
            + pn.scale_color_ordinal()
            + pn.geom_line()
            + pn.geom_point(size=2)
            + pn.scale_color_brewer(type="qual", palette="Set2")  # type: ignore
            + pn.labs(y="AUROC", x="Year of evaluation", color="Final year of training")
            + THEME
            + pn.theme(
                panel_grid_major=pn.element_line(color="lightgrey", size=0.25, linetype="dotted"),
                axis_text_x=pn.element_text(size=10),
                legend_position=(0.5, 0.2),
            )
        )
        if not self.relative_time:
            plot = plot + pn.scale_x_datetime(labels=format_datetime, date_breaks="1 year")  # noqa: ERA001

        return plot


if __name__ == "__main__":
    coloredlogs.install(level="INFO", fmt="%(asctime)s [%(levelname)s] %(message)s")

    for feature_set in ["structured_only", "text_only", "structured_text"]:
        runs = MlflowClientWrapper().get_runs_from_experiment(
            f"ECT-trunc-and-hp-{feature_set}-xgboost-no-lookbehind-filter_best_run_temporal_eval"
        )
        performances = temporal_stability(runs)
        for is_relative_time in [True, False]:
            relative_time_str = "relative" if is_relative_time else "absolute"
            plot = TemporalStabilityPlot(performances, relative_time=is_relative_time)()
            save_dir = Path("E:/shared_resources/ect/eval_runs/figures")
            save_dir.mkdir(parents=True, exist_ok=True)
            plot.save(
                save_dir / f"temporal_stability_{feature_set}_{relative_time_str}.png", dpi=300
            )
