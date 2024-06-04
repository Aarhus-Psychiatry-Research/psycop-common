# %%
# %load_ext autoreload
# %autoreload 2

# %%
import datetime

from psycop.projects.cvd.cohort_examination.label_by_outcome_type import label_by_outcome_type
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    get_first_cvd_indicator,
)

# %%
df_lab_result = get_first_cvd_indicator()

# %%
import plotnine as pn
import polars as pl

from psycop.projects.t2d.paper_outputs.config import COLORS, FONT_SIZES, THEME

# %%
grouped_by_outcome = label_by_outcome_type(pl.from_pandas(df_lab_result), group_col="cause")

# %%
grouped_by_outcome.filter(pl.col("outcome_type").is_null())

# %%
p = (
    pn.ggplot(grouped_by_outcome.fill_null("Unknown"), pn.aes(x="timestamp", fill="outcome_type"))
    # + pn.geom_density(alpha=0.7)
    + pn.geom_histogram()
    + pn.geom_vline(xintercept=datetime.datetime(2013, 1, 1), color="black")
    + pn.theme_minimal()
    + pn.xlab("Date")
    + pn.ylab("Incident CVD in month")
    + THEME
    + pn.scale_x_datetime(limits=(datetime.datetime(2011, 1, 1), datetime.datetime(2021, 11, 22)))
    + pn.theme(
        axis_text_x=pn.element_text(size=FONT_SIZES.axis_tick_labels, angle=45, hjust=1),
        axis_text_y=pn.element_text(size=FONT_SIZES.axis_tick_labels),
    )
    + pn.facet_wrap(facets="outcome_type", scales="free")
)

p

# %%
from psycop.projects.t2d.paper_outputs.selected_runs import get_best_eval_pipeline

p.save("Incident_CVD_by_time.png", dpi=600, width=15, height=10)

# %%
