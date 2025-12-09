import datetime

import plotnine as pn

from psycop.projects.t2d.paper_outputs.config import COLORS, FONT_SIZES, THEME
from psycop.projects.t2d_bigdata.cohort_examination.incidence_by_time.model import (
    IncidenceByTimeModel,
)


def incidence_by_time_distribution(
    model: IncidenceByTimeModel,
    limits: tuple[datetime.datetime, datetime.datetime] = (
        datetime.datetime(2011, 1, 1),
        datetime.datetime(2021, 11, 22),
    ),
) -> pn.ggplot:
    distribution_plot = (
        pn.ggplot(model, pn.aes(x="timestamp", y=pn.after_stat("count"), fill="outcome_type"))
        + THEME
        + pn.scale_x_datetime(limits=limits)
        + pn.geom_vline(xintercept=datetime.datetime(2013, 1, 1), color="black")
        + pn.xlab("")
        + pn.theme(
            axis_text_x=pn.element_text(size=FONT_SIZES.axis_tick_labels, angle=45, hjust=1),
            axis_text_y=pn.element_text(size=FONT_SIZES.axis_tick_labels),
        )
        + pn.scale_fill_manual(COLORS.color_scale())
        + pn.geom_density(position="fill", alpha=0.7)
        + pn.ylab("Incident T2D")
        + pn.geom_vline(xintercept=datetime.datetime(2013, 1, 1), color="black", linetype="dashed")
    )
    return distribution_plot


def incidence_by_time_faceted(
    model: IncidenceByTimeModel,
    limits: tuple[datetime.datetime, datetime.datetime] = (
        datetime.datetime(2011, 1, 1),
        datetime.datetime(2021, 11, 22),
    ),
) -> pn.ggplot:
    facet_plot = (
        pn.ggplot(model, pn.aes(x="timestamp", fill="outcome_type", y=pn.after_stat("count")))
        + pn.xlab("")
        + THEME
        + pn.scale_x_datetime(limits=limits)
        + pn.theme(
            axis_text_x=pn.element_text(size=FONT_SIZES.axis_tick_labels, angle=45, hjust=1),
            axis_text_y=pn.element_blank(),
            axis_ticks_major_y=pn.element_blank(),
        )
        + pn.ylab("")
        + pn.scale_fill_manual(COLORS.color_scale())
        + pn.geom_density(alpha=0.7)
        + pn.theme(legend_position="none")
        + pn.facet_wrap(facets="outcome_type", scales="free")
        + pn.geom_vline(xintercept=datetime.datetime(2013, 1, 1), color="black", linetype="dashed")
    )
    return facet_plot
