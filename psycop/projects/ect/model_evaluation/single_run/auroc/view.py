import logging
from dataclasses import dataclass

import plotnine as pn

from psycop.projects.ect.model_evaluation.single_run.auroc.model import AUROC
from psycop.projects.ect.model_evaluation.single_run.single_run_artifact import SingleRunPlot
from psycop.projects.t2d.paper_outputs.config import THEME


@dataclass(frozen=True)
class AUROCPlot(SingleRunPlot):
    data: AUROC

    def __call__(self) -> pn.ggplot:
        logging.info(f"Starting {self.__class__.__name__}")
        auroc_label = pn.annotate(
            "text",
            label=f"AUROC (95% CI): {self.data.mean:.2f} ({self.data.ci[0]:.2f}-{self.data.ci[1]:.2f})",
            x=1,
            y=0,
            ha="right",
            va="bottom",
            size=10,
        )

        # Plot AUC ROC curve
        return (
            pn.ggplot(self.data.to_dataframe(), pn.aes(x="fpr", y="tpr"))
            + pn.geom_path(size=1)
            + pn.geom_line(pn.aes(y="tpr_upper"), linetype="dashed", color="grey")
            + pn.geom_line(pn.aes(y="tpr_lower"), linetype="dashed", color="grey")
            + pn.labs(x="1 - Specificity", y="Sensitivity")
            + pn.xlim(0, 1)
            + pn.ylim(0, 1)
            + pn.geom_abline(intercept=0, slope=1, linetype="dotted")
            + auroc_label
            + THEME
        )
