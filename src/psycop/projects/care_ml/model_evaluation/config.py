from dataclasses import dataclass
from pathlib import Path

import plotnine as pn
from care_ml.utils.best_runs import Run, RunGroup


@dataclass
class BestRun:
    wandb_group: str
    model: str
    pos_rate: float


EVALUATION_ROOT = Path(__file__).parent

POS_RATE = 0.05

# Best model on structured features
DEV_GROUP_NAME = "exophthalmia-intombed"
DEVELOPMENT_GROUP = RunGroup(name=DEV_GROUP_NAME)
BEST_DEV_RUN = Run(
    group=DEVELOPMENT_GROUP,
    name="nonrandomnessseparableness",
    pos_rate=POS_RATE,
)

EVAL_GROUP_NAME = f"{DEV_GROUP_NAME}-eval-on-test"
EVAL_GROUP = RunGroup(name=EVAL_GROUP_NAME)
EVAL_RUN = Run(
    group=EVAL_GROUP,
    name="unaugmentativepreconcurrently",
    pos_rate=POS_RATE,
)


# Best model on structured features + text features
TEXT_DEV_GROUP_NAME = "seigneurial-normalacy"
TEXT_DEVELOPMENT_GROUP = RunGroup(name=TEXT_DEV_GROUP_NAME)
TEXT_BEST_DEV_RUN = Run(
    group=TEXT_DEVELOPMENT_GROUP,
    name="coucheporringer",
    pos_rate=POS_RATE,
)

TEXT_EVAL_GROUP_NAME = f"{TEXT_DEV_GROUP_NAME}-eval-on-test"
TEXT_EVAL_GROUP = RunGroup(name=TEXT_EVAL_GROUP_NAME)
TEXT_EVAL_RUN = Run(
    group=TEXT_EVAL_GROUP,
    name="finchautocrator",  # should be changed when we run on "test"
    pos_rate=POS_RATE,
)


# output paths
GENERAL_ARTIFACT_PATH = (
    EVALUATION_ROOT
    / "outputs_for_publishing"
    / f"{EVAL_GROUP.name}"
    / f"{EVAL_RUN.name}"
)
FIGURES_PATH = GENERAL_ARTIFACT_PATH / "figures"
TABLES_PATH = GENERAL_ARTIFACT_PATH / "tables"
ESTIMATES_PATH = GENERAL_ARTIFACT_PATH / "estimates"
ROBUSTNESS_PATH = FIGURES_PATH / "robustness"

TEXT_GENERAL_ARTIFACT_PATH = (
    EVALUATION_ROOT
    / "outputs_for_publishing"
    / f"{TEXT_EVAL_GROUP.name}"
    / f"{TEXT_EVAL_RUN.name}"
)
TEXT_FIGURES_PATH = TEXT_GENERAL_ARTIFACT_PATH / "figures"
TEXT_TABLES_PATH = TEXT_GENERAL_ARTIFACT_PATH / "tables"
TEXT_ESTIMATES_PATH = TEXT_GENERAL_ARTIFACT_PATH / "estimates"
TEXT_ROBUSTNESS_PATH = TEXT_FIGURES_PATH / "robustness"

for path in [
    GENERAL_ARTIFACT_PATH,
    FIGURES_PATH,
    TABLES_PATH,
    ESTIMATES_PATH,
    ROBUSTNESS_PATH,
    TEXT_GENERAL_ARTIFACT_PATH,
    TEXT_FIGURES_PATH,
    TEXT_TABLES_PATH,
    TEXT_ESTIMATES_PATH,
    TEXT_ROBUSTNESS_PATH,
]:
    path.mkdir(exist_ok=True, parents=True)


# @dataclass
class OutputMapping:
    coercion_incidence_by_time: str = "eFigure 3"
    shap_table: str = "eTable 3"
    shap_plots: str = "Figure 3"


OUTPUT_MAPPING = OutputMapping()

PN_THEME = pn.theme_bw() + pn.theme(
    panel_grid=pn.element_blank(),
    text=(pn.element_text(family="Times New Roman")),
    axis_text=pn.element_text(size=15),
    axis_title=pn.element_text(size=18),
    plot_title=pn.element_text(size=22),
    dpi=300,
)

COLOURS = {
    "blue": "#7B9EBD",
    "green": "#AECAAE",
    "red": "#BD354C",
    "yellow": "#E8D992",
    "purple": "#684D82",
}

MODEL_NAME = {
    "unaugmentativepreconcurrently": "Baseline Model",
    "finchautocrator": "Text-Enhanced Model",
}
