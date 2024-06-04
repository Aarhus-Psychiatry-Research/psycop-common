# %%
# %load_ext autoreload
# %autoreload 2

# %%
import datetime

from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    get_first_cvd_indicator,
)

# %%
df_lab_result = get_first_cvd_indicator()

# %%
import pandas as pd
import plotnine as pn

from psycop.projects.t2d.paper_outputs.config import COLORS, FONT_SIZES, THEME

# %%
p = (
    pn.ggplot(df_lab_result, pn.aes(x="timestamp"))
    + pn.geom_density(alpha=0.8, fill=COLORS.primary)
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
)

p

# %%
from psycop.projects.t2d.paper_outputs.selected_runs import get_best_eval_pipeline

p.save("Incident_CVD_by_time.png", dpi=600)

# %%
