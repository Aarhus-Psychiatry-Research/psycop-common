# %%
# %load_ext autoreload
# %autoreload 2

# %%
import datetime
from importlib.metadata import distribution

from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    get_first_cvd_indicator,
)

# %%
df_lab_result = get_first_cvd_indicator()

# %%
import plotnine as pn
import polars as pl

# %%
from psycop.projects.cvd.cohort_examination.label_by_outcome_type import label_by_outcome_type
from psycop.projects.t2d.paper_outputs.config import COLORS, FONT_SIZES, THEME

grouped_by_outcome = label_by_outcome_type(
    pl.from_pandas(df_lab_result), group_col="cause"
).with_columns(
    pl.when(pl.col("outcome_type").str.contains("artery"))
    .then(pl.lit("PAD"))
    .otherwise("outcome_type")
    .alias("outcome_type")
)

# %%
from psycop.projects.cvd.feature_generation.cohort_definition.eligible_prediction_times.single_filters import (
    CVDWashoutMove,
)

filtered_after_move = CVDWashoutMove().apply(grouped_by_outcome.lazy()).collect()

# %%
limits = (datetime.datetime(2011, 1, 1), datetime.datetime(2021, 11, 22))
facet_plot = (
    pn.ggplot(
        filtered_after_move, pn.aes(x="timestamp", fill="outcome_type", y=pn.after_stat("count"))
    )
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
facet_plot


# %%
distribution_plot = (
    pn.ggplot(
        filtered_after_move, pn.aes(x="timestamp", y=pn.after_stat("count"), fill="outcome_type")
    )
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
    + pn.ylab("Incident CVD")
    + pn.geom_vline(xintercept=datetime.datetime(2013, 1, 1), color="black", linetype="dashed")
)

distribution_plot


# %%
from psycop.projects.t2d.paper_outputs.selected_runs import get_best_eval_pipeline

p.save("Incident_CVD_by_time.png", dpi=600, width=15, height=10)

# %%
