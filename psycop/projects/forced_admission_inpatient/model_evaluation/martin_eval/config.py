from dataclasses import dataclass
from datetime import datetime

import plotnine as pn

from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import RunGroup

########################################
# UPDATE THESE TO SELECT MODEL OUTPUTS #
########################################
DEV_GROUP_NAME = "physiurgic-letterleaf"
DEVELOPMENT_GROUP = RunGroup(name=DEV_GROUP_NAME)
BEST_POS_RATE = 0.05
BEST_RUN_NAME = DEVELOPMENT_GROUP.get_best_runs_by_lookahead()[0, 2]

EVAL_GROUP_NAME = f"{DEV_GROUP_NAME}-eval-on-test"
EVAL_GROUP = RunGroup(name=EVAL_GROUP_NAME)

################
# OUTPUT PATHS #
################
date_str = datetime.now().strftime("%Y-%m-%d")

FA_PN_THEME = pn.theme_bw() + pn.theme(
    panel_grid=pn.element_blank(),
    axis_title=pn.element_text(size=14),
)


@dataclass
class Colors:
    primary = "#0072B2"
    secondary = "#009E73"
    tertiary = "#D55E00"
    background = "lightgray"


COLORS = Colors()
