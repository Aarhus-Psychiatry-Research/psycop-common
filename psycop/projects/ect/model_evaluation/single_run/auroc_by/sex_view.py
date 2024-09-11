import logging
from dataclasses import dataclass

import plotnine as pn
import polars as pl

from psycop.projects.ect.model_evaluation.single_run.auroc_by.auroc_by_view import auroc_by_view
from psycop.projects.ect.model_evaluation.single_run.auroc_by.sex_model import AurocBySexDF
from psycop.projects.ect.model_evaluation.single_run.single_run_artifact import SingleRunPlot

log = logging.getLogger(__file__)


@dataclass
class AUROCBySex(SingleRunPlot):
    data: AurocBySexDF

    def __call__(self) -> pn.ggplot:
        log.info(f"Starting {self.__class__.__name__}")

        data = self.data.with_columns(
            pl.when(pl.col("sex")).then(pl.lit("Female")).otherwise(pl.lit("Male")).alias("sex")
        )

        return auroc_by_view(
            df=data.to_pandas(), x_column="sex", line_y_col_name="auroc", xlab="Sex"
        )
