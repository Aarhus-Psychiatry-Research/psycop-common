import logging
from dataclasses import dataclass

import plotnine as pn

from psycop.projects.cvd.model_evaluation.single_run.auroc_by.auroc_by_view import auroc_by_view
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.sex_model import AurocBySexDF
from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import SingleRunPlot


@dataclass
class SexByAUROC(SingleRunPlot):
    data: AurocBySexDF

    def __call__(self) -> pn.ggplot:
        logging.info(f"Starting {self.__class__.__name__}")

        return auroc_by_view(
            df=self.data.to_pandas(), x_column="sex", line_y_col_name="auroc", xlab="Sex"
        )
