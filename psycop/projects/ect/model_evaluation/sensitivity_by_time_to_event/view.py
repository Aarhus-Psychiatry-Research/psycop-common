from dataclasses import dataclass

import plotnine as pn

from psycop.projects.ect.model_evaluation.single_run_artifact import SingleRunPlot
from psycop.projects.t2d.paper_outputs.config import FONT_SIZES, THEME, ColorsPTC

from .model import SensitivityByTTEDF


@dataclass(frozen=True)
class SensitivityByTTEPlot(SingleRunPlot):
    colors: ColorsPTC
    outcome_label: str
    data: SensitivityByTTEDF
    desired_positive_rate: float = 0.05

    def __call__(self) -> pn.ggplot:
        df = self.data.to_pandas()
        categories = df["unit_from_event_binned"].dtype.categories  # type: ignore
        df["unit_from_event_binned"] = df["unit_from_event_binned"].cat.set_categories(
            new_categories=categories,
            ordered=True,  # type: ignore
        )

        p = (
            pn.ggplot(
                df,
                pn.aes(
                    x="unit_from_event_binned",
                    y="sensitivity",
                    ymin="ci_lower",
                    ymax="ci_upper",
                    color="actual_positive_rate",
                    group="actual_positive_rate",
                ),
            )
            + pn.scale_x_discrete()
            + pn.expand_limits(y=0)
            + pn.geom_path(size=0.5)
            + pn.geom_point(size=1)
            + pn.geom_errorbar(width=0.1)
            + pn.labs(x="Days", y="Sensitivity", color="Predicted positive rate")
            + pn.scale_color_brewer(type="qual", palette="Set2")  # type: ignore
            + THEME
            + pn.theme(
                axis_text_x=pn.element_text(size=FONT_SIZES.axis_tick_labels, angle=45, hjust=1),
                axis_text_y=pn.element_text(size=FONT_SIZES.axis_tick_labels),
                legend_position=(0.8, 0.8),

            )
        )

        for value in df["actual_positive_rate"].unique():
            p += pn.geom_path(df[df["actual_positive_rate"] == value], group=1)

        return p
