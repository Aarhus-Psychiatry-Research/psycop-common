from dataclasses import dataclass

import plotnine as pn
import polars as pl

from psycop.projects.ect.model_evaluation.auroc_by.roc_by_multiple_runs_model import ROCByGroupDF
from psycop.projects.ect.model_evaluation.single_run_artifact import SingleRunPlot


@dataclass(frozen=True)
class ROCByGroupPlot(SingleRunPlot):
    data: ROCByGroupDF

    def __call__(self) -> pn.ggplot:
        abline_df = pl.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})

        df = self.data.with_columns(
            pl.concat_str(pl.col("run_name"), pl.col("AUROC").round(3), separator=": AUROC=")
        )
        order = (
            df.group_by("run_name")
            .agg(pl.col("AUROC").max())
            .sort(by="AUROC", descending=True)
            .get_column("run_name")
            .to_list()
        )
        df = df.with_columns(pl.col("run_name").cast(pl.Enum(order)))

        return (
            pn.ggplot(df, pn.aes(x="fpr", y="tpr", color="run_name"))
            + pn.geom_line()
            + pn.geom_line(data=abline_df, linetype="dashed", color="grey", alpha=0.5)
            + pn.labs(x="1 - Specificity", y="Sensitivty")
            + pn.coord_cartesian(xlim=(0, 1), ylim=(0, 1))
            + pn.theme_classic()
            + pn.theme(
                legend_position=(0.65, 0.25),
                legend_direction="vertical",
                legend_title=pn.element_blank(),
                axis_title=pn.element_text(size=14),
                legend_text=pn.element_text(size=11),
                axis_text=pn.element_text(size=10),
                figure_size=(5, 5),
            )
        )
