import logging
from dataclasses import dataclass

import plotnine as pn

from psycop.projects.cvd.model_evaluation.single_run.auroc.model import AUROC
from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import SingleRunPlot


@dataclass(frozen=True)
class AUROCPlot(SingleRunPlot):
    data: AUROC
    title: str = "Receiver Operating Characteristic (ROC) Curve"

    def __call__(self) -> pn.ggplot:
        logging.info(f"Starting {self.__class__.__name__}")
        auroc_label = pn.annotate(
            "text",
            label=f"AUROC (95% CI): {self.data.mean:.3f} ({self.data.ci[0]:.3f}-{self.data.ci[1]:.3f})",
            x=1,
            y=0,
            ha="right",
            va="bottom",
            size=10,
        )

        # Plot AUC ROC curve
        return (
            pn.ggplot(self.data.to_dataframe(), pn.aes(x="fpr", y="tpr"))
            + pn.geom_line(size=1)
            + pn.geom_line(pn.aes(y="tpr_upper"), linetype="dashed", color="grey")
            + pn.geom_line(pn.aes(y="tpr_lower"), linetype="dashed", color="grey")
            + pn.labs(title=self.title, x="1 - Specificity", y="Sensitivity")
            + pn.xlim(0, 1)
            + pn.ylim(0, 1)
            + pn.geom_abline(intercept=0, slope=1, linetype="dotted")
            + pn.theme(
                axis_text=pn.element_text(size=10, weight="bold", color="black"),
                axis_title=pn.element_text(size=14, color="black"),
            )
            + auroc_label
        )
