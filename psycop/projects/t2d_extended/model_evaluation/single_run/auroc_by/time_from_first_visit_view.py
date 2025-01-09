import logging
from dataclasses import dataclass

import plotnine as pn

from psycop.projects.t2d_extended.model_evaluation.single_run.auroc_by.auroc_by_view import auroc_by_view
from psycop.projects.t2d_extended.model_evaluation.single_run.auroc_by.time_from_first_visit_model import (
    TimeFromFirstVisitDF,
)
from psycop.projects.t2d_extended.model_evaluation.single_run.single_run_artifact import SingleRunPlot


@dataclass
class AUROCByTimeFromFirstVisitPlot(SingleRunPlot):
    data: TimeFromFirstVisitDF

    def __call__(self) -> pn.ggplot:
        logging.info(f"Starting {self.__class__.__name__}")

        return auroc_by_view(
            df=self.data.to_pandas(),
            x_column="unit_from_event_binned",
            line_y_col_name="auroc",
            xlab="Months from first visit",
        )
