import logging

import plotnine as pn

from psycop.projects.cvd.model_evaluation.single_run.auroc_by.auroc_by_view import auroc_by_view
from psycop.projects.cvd.model_evaluation.single_run.auroc_by.day_of_week_model import (
    AUROCByDayOfWeekDF,
)
from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import SingleRunPlot


class AUROCByDayOfWeek(SingleRunPlot):
    data: AUROCByDayOfWeekDF

    def __call__(self) -> pn.ggplot:
        logging.info(f"Starting {self.__class__.__name__}")

        return auroc_by_view(
            df=self.data.to_pandas(), x_column="time_bin", line_y_col_name="auroc", xlab="Age"
        )
