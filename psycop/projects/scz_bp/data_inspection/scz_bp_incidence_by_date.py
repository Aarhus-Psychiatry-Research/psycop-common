# load outcome timestamps
import datetime

import pandas as pd
import plotnine as pn
import polars as pl

from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.single_filters import (
    SczBpWashoutMoveFilter,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.bp_diagnoses import (
    get_first_bp_diagnosis,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.scz_diagnoses import (
    get_first_scz_diagnosis,
)

dfs = {"scz": get_first_scz_diagnosis(), "bp": get_first_bp_diagnosis()}
for df_type in dfs:
    dfs[df_type]["source"] = df_type

df = pd.concat(list(dfs.values()))

df = SczBpWashoutMoveFilter().apply(pl.from_pandas(df).lazy()).collect()
(
    pn.ggplot(df, pn.aes(x="timestamp", fill="source"))
    + pn.geom_histogram()
    + pn.geom_vline(xintercept=datetime.datetime(2013, 1, 1), color="black")
    + pn.theme_minimal()
    + pn.xlab("Date")
    + pn.ylab("N Incident BP/SCZ")
    + pn.theme_minimal()
    + pn.theme(
        axis_text_x=pn.element_text(size=12, angle=45, hjust=1),
        axis_text_y=pn.element_text(size=12),
        legend_position="none",
    )
    + pn.scale_x_datetime(limits=(datetime.datetime(2011, 1, 1), datetime.datetime(2021, 11, 22)))
    + pn.facet_wrap(facets="source", scales="free")
)
