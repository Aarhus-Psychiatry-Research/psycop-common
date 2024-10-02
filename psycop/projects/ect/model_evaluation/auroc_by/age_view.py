import logging

import plotnine as pn
from attr import dataclass

from psycop.projects.ect.model_evaluation.auroc_by.age_model import AUROCByAgeDF
from psycop.projects.ect.model_evaluation.auroc_by.auroc_by_view import auroc_by_view
from psycop.projects.ect.model_evaluation.single_run_artifact import SingleRunPlot


@dataclass
class AUROCByAge(SingleRunPlot):
    data: AUROCByAgeDF

    def __call__(self) -> pn.ggplot:
        logging.info(f"Starting {self.__class__.__name__}")

        return auroc_by_view(
            df=self.data.to_pandas(), x_column="age_binned", line_y_col_name="auroc", xlab="Age"
        )
