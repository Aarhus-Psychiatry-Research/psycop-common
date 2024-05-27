from dataclasses import dataclass
from turtle import color

import plotnine as pn

from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import SingleRunPlot
from psycop.projects.t2d.paper_outputs.config import COLORS, FONT_SIZES, THEME, ColorsPTC

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
            + pn.geom_path(group=1, size=0.5)
            + pn.geom_point(size=1)
            + pn.geom_errorbar(width=0.1)
            + pn.labs(x="Months to outcome", y="Sensitivity")
            + THEME
            + pn.theme(
                axis_text_x=pn.element_text(size = FONT_SIZES.axis_tick_labels, angle=45, hjust=1),
                axis_text_y=pn.element_text(size = FONT_SIZES.axis_tick_labels)
            )
        )

        for value in df["actual_positive_rate"].unique():
            p += pn.geom_path(df[df["actual_positive_rate"] == value], group=1)

        return p
