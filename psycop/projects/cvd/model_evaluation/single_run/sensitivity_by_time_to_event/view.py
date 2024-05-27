from dataclasses import dataclass
from turtle import color

import plotnine as pn

from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import SingleRunPlot
from psycop.projects.t2d.paper_outputs.config import COLORS, ColorsPTC

from .model import SensitivityByTTEDF


@dataclass(frozen=True)
class SensitivityByTTEPlot(SingleRunPlot):
    colors: ColorsPTC
    outcome_label: str
    data: SensitivityByTTEDF
    desired_positive_rate: float = 0.05

    def __call__(self) -> pn.ggplot:
        df = self.data.to_pandas()
        categories = df["unit_from_event_binned"].dtype.categories[::-1]  # type: ignore
        df["unit_from_event_binned"] = df["unit_from_event_binned"].cat.set_categories(
            new_categories=categories,
            ordered=True,  # type: ignore
        )

        p = (
            pn.ggplot(
                df,
                pn.aes(
                    x="unit_from_event_binned", y="sensitivity", ymin="ci_lower", ymax="ci_upper"
                ),
            )
            + pn.scale_x_discrete(reverse=True)
            + pn.geom_path(fill=self.colors.primary, size=1)
            + pn.geom_point(color=self.colors.primary, size=1)
            + pn.geom_errorbar(color=self.colors.primary, width=0.1)
            + pn.labs(x="Months to outcome", y="Sensitivity")
            + pn.scale_color_manual([self.colors.primary])
        )

        for value in df["actual_positive_rate"].unique():
            p += pn.geom_path(df[df["actual_positive_rate"] == value], group=1)

        return p
