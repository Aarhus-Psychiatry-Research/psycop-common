from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import plotnine as pn

########################################
# UPDATE THESE TO SELECT MODEL OUTPUTS #
########################################
MODEL_ALGORITHM = 1  # 0 fo logistic regression and 1 for best xgboost

CLOZAPINE_EVAL_OUTPUT_DIR = Path("E:/shared_resources/clozapine/eval/")

################
# OUTPUT PATHS #
################
date_str = datetime.now().strftime("%Y-%m-%d")

FA_PN_THEME = pn.theme_bw() + pn.theme(
    panel_grid=pn.element_blank(), axis_title=pn.element_text(size=14)
)


@dataclass
class Colors:
    primary = "#0072B2"
    secondary = "#009E73"
    tertiary = "#D55E00"
    bg_primary = "lightgray"
    bg_secondary = "darkgray"


COLORS = Colors()
