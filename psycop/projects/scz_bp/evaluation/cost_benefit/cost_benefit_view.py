from collections.abc import Sequence

import numpy as np
import plotnine as pn

from psycop.projects.scz_bp.evaluation.cost_benefit.cost_benefit_model import (
    CostBenefitDistributions,
    CostBenefitDistributionsDF,
)


def plot_sampling_distribution(df: CostBenefitDistributionsDF, col_to_plot: str) -> pn.ggplot:
    # Create string for plottign for col name by removing underscores and capitalizing
    col_to_plot_str = col_to_plot.replace("_", " ").capitalize()

    p = (
        pn.ggplot(df, pn.aes(x=col_to_plot))
        + pn.geom_histogram(
            pn.aes(y=pn.after_stat("density")), bins=30, fill="#009E73", alpha=0.4, color="#009E73"
        )
        + pn.geom_density(color="#D55E00", size=1.5)
        + pn.labs(x=f"{col_to_plot_str}", y="Density")
        + pn.theme_classic()
        + pn.theme(panel_grid_major=pn.element_blank(), panel_grid_minor=pn.element_blank())
    )
    # if col_to_plot == "cost_benefit_ratio" then add vertical lines for the 5th, and 95th percentiles, the mean and the median
    if col_to_plot == "cost_benefit_ratio":
        p += pn.geom_vline(
            xintercept=int(df[col_to_plot].median()),
            linetype="dotted",
            color="red",  # type: ignore
        )
        p += pn.annotate(
            "text",
            x=int(df[col_to_plot].median()),  # type: ignore
            y=0.028,
            label="Median",
            color="red",
            va="center",
            ha="right",
            size=6,
            angle=90,
        )
        p += pn.geom_vline(
            xintercept=int(df[col_to_plot].mean()),
            linetype="dotted",
            color="blue",  # type: ignore
        )
        p += pn.annotate(
            "text",
            x=int(df[col_to_plot].mean()),  # type: ignore
            y=0.028,
            label="Mean",
            color="blue",
            va="center",
            ha="right",
            size=6,
            angle=90,
        )
        p += pn.geom_vline(
            xintercept=int(np.percentile(df[col_to_plot], 5)), linetype="dotted", color="green"
        )
        p += pn.annotate(
            "text",
            x=int(np.percentile(df[col_to_plot], 5)),
            y=0.028,
            label="5th perc.",
            color="green",
            va="center",
            ha="right",
            size=6,
            angle=90,
        )
        p += pn.geom_vline(
            xintercept=int(np.percentile(df[col_to_plot], 95)), linetype="dotted", color="green"
        )
        p += pn.annotate(
            "text",
            x=int(np.percentile(df[col_to_plot], 95)),
            y=0.028,
            label="95th perc.",
            color="green",
            va="center",
            ha="right",
            size=6,
            angle=90,
        )

    return p


def plot_sampling_distributions(
    cost_benefit_distributions: CostBenefitDistributions,
) -> Sequence[pn.ggplot]:
    df = cost_benefit_distributions.get_dataframe()
    col_names = df.columns

    return [plot_sampling_distribution(df=df, col_to_plot=col_to_plot) for col_to_plot in col_names]
