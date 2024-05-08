from dataclasses import dataclass
from pathlib import Path

import plotnine as pn


T2D_PN_THEME = pn.theme_bw() + pn.theme(
    panel_grid=pn.element_blank(), axis_title=pn.element_text(size=14)
)

SCZ_BP_EVAL_OUTPUT_DIR = Path(__file__).parent / "outputs"

@dataclass
class Colors:
    primary = "#0072B2"
    secondary = "#009E73"
    tertiary = "#D55E00"
    background = "lightgray"


COLORS = Colors()
