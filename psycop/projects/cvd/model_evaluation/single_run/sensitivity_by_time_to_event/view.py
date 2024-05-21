from dataclasses import dataclass

import plotnine as pn

from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import SingleRunPlot

from .model import SensitivityByTTEDF


@dataclass(frozen=True)
class SensitivityByTTEPlot(SingleRunPlot):
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
                    x="unit_from_event_binned",
                    y="sensitivity",
                    ymin="ci_lower",
                    ymax="ci_upper",
                    color="actual_positive_rate",
                ),
            )
            + pn.scale_x_discrete(reverse=True)
            + pn.geom_point()
            + pn.geom_linerange(size=0.5)
            + pn.labs(x="Months to outcome", y="Sensitivity")
            + pn.theme(axis_text_x=pn.element_text(rotation=45, hjust=1))
            + pn.scale_color_brewer(type="qual", palette=2)
            + pn.labs(color="Predicted Positive Rate")
            + pn.theme(
                panel_grid_major=pn.element_blank(),
                panel_grid_minor=pn.element_blank(),
                legend_position=(0.3, 0.88),
            )
        )

        for value in df["actual_positive_rate"].unique():
            p += pn.geom_path(df[df["actual_positive_rate"] == value], group=1)

        return p
