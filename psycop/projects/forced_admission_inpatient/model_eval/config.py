from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import plotnine as pn

from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import RunGroup

########################################
# UPDATE THESE TO SELECT MODEL OUTPUTS #
########################################
MODEL_NAME = "full_model_with_text_features"
PROJECT_MODEL_DIR = Path(
    f"E:/shared_resources/forced_admissions_inpatient/models/{MODEL_NAME}/pipeline_eval",
)
MODEL_ALGORITHM = 1  # 0 fo logistic regression and 1 for best xgboost

DEV_GROUP_NAME = "undertided-acetanilide"
DEVELOPMENT_GROUP = RunGroup(model_name=MODEL_NAME, group_name=DEV_GROUP_NAME)
BEST_POS_RATE = 0.04


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
