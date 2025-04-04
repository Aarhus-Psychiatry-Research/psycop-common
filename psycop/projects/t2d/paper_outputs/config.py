from dataclasses import dataclass
from datetime import datetime

import plotnine as pn

from psycop.projects.t2d.utils.pipeline_objects import RunGroup

########################################
# UPDATE THESE TO SELECT MODEL OUTPUTS #
########################################
DEV_GROUP_NAME = "urosepsis-helicoid"
DEVELOPMENT_GROUP = RunGroup(name=DEV_GROUP_NAME)
BEST_POS_RATE = 0.03

EVAL_GROUP_NAME = f"{DEV_GROUP_NAME}-eval-on-test"
EVAL_GROUP = RunGroup(name=EVAL_GROUP_NAME)

################
# OUTPUT PATHS #
################
date_str = datetime.now().strftime("%Y-%m-%d")

T2D_PN_THEME = pn.theme_bw() + pn.theme(
    panel_grid=pn.element_blank(), axis_title=pn.element_text(size=14)
)

from collections.abc import Sequence
from typing import Protocol


class ColorsPTC(Protocol):
    primary: str
    secondary: str
    tertiary: str
    background: str


@dataclass(frozen=True)
class Colors(ColorsPTC):
    primary = "#0072B2"
    secondary = "#01611E"
    tertiary = "#B05B00"
    quarternary = "#570018"
    pentary = "darkgray"
    background = "lightgray"

    def color_scale(self) -> Sequence[str]:
        return [getattr(self, attr) for attr in dir(self) if attr.endswith("ary")]


@dataclass(frozen=True)
class FontSizes:
    axis_tick_labels: int = 12


COLORS = Colors()
FONT_SIZES = FontSizes()
THEME = pn.theme_classic() + pn.theme(
    axis_text_x=pn.element_text(size=FONT_SIZES.axis_tick_labels),
    axis_text_y=pn.element_text(size=FONT_SIZES.axis_tick_labels),
)
