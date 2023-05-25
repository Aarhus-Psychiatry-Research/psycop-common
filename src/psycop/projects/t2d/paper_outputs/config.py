from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import plotnine as pn
from psycop.projects.t2d.utils.best_runs import PipelineRun, RunGroup

########################################
# UPDATE THESE TO SELECT MODEL OUTPUTS #
########################################
DEV_GROUP_NAME = "mistouching-unwontedness"
DEVELOPMENT_GROUP = RunGroup(name=DEV_GROUP_NAME)
BEST_POS_RATE = 0.03
BEST_DEV_PIPELINE = PipelineRun(
    group=DEVELOPMENT_GROUP,
    name="surefootedlygoatpox",
    pos_rate=BEST_POS_RATE,
)

EVAL_GROUP_NAME = f"{DEV_GROUP_NAME}-eval-on-test"
EVAL_GROUP = RunGroup(name=EVAL_GROUP_NAME)
BEST_EVAL_PIPELINE = PipelineRun(
    group=EVAL_GROUP,
    name="pseudoreformatoryhizz",
    pos_rate=BEST_POS_RATE,
)


################
# OUTPUT PATHS #
################
date_str = datetime.now().strftime("%Y-%m-%d")

PN_THEME = pn.theme_bw() + pn.theme(panel_grid=pn.element_blank())


@dataclass
class Colors:
    primary = "#0072B2"
    secondary = "#009E73"
    tertiary = "#D55E00"
    background = "lightgray"


COLORS = Colors()
