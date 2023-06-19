from dataclasses import dataclass
from pathlib import Path

import plotnine as pn

from psycop.projects.forced_admission_inpatient.utils.best_runs import Run, RunGroup


@dataclass
class BestRun:
    wandb_group: str
    model: str
    pos_rate: float


PROJECT_MODEL_DIR = Path(
    "E:\\shared_resources\\forced_admissions_inpatient\\pipeline_eval",
)
POS_RATE = 0.05


DEV_GROUP_NAME = "mangi-bouquets"

DEVELOPMENT_GROUP = RunGroup(name=DEV_GROUP_NAME)

BEST_RUN_NAME = DEVELOPMENT_GROUP.get_best_runs_by_lookahead()[1, 2]

BEST_DEV_RUN = Run(
    group=DEVELOPMENT_GROUP,
    name=BEST_RUN_NAME,
    pos_rate=POS_RATE,
)

EVAL_GROUP_NAME = f"{DEV_GROUP_NAME}-eval-on-test"
EVAL_GROUP = RunGroup(name=EVAL_GROUP_NAME)
EVAL_RUN = Run(
    group=EVAL_GROUP,
    name="congreganistcarburating",
    pos_rate=POS_RATE,
)

EVAL_CONFIG = EVAL_RUN.cfg

EVAL_ROOT = EVAL_CONFIG.project.project_path / "pipeline_eval"

# output paths
GENERAL_ARTIFACT_PATH = EVAL_ROOT / f"{EVAL_GROUP.name}" / f"{EVAL_RUN.name}"
FIGURES_PATH = GENERAL_ARTIFACT_PATH / "figures"
TABLES_PATH = GENERAL_ARTIFACT_PATH / "tables"
ESTIMATES_PATH = GENERAL_ARTIFACT_PATH / "estimates"
ROBUSTNESS_PATH = FIGURES_PATH / "robustness"


for path in [
    GENERAL_ARTIFACT_PATH,
    FIGURES_PATH,
    TABLES_PATH,
    ESTIMATES_PATH,
    ROBUSTNESS_PATH,
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
    "patteredauxograph": "Baseline Model",
}
