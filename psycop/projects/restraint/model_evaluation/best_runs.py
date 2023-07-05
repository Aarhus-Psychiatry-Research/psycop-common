from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class BestRun:
    wandb_group: str
    model: str
    pos_rate: float


best_run = BestRun(
    wandb_group="exophthalmia-intombed",
    model="nonrandomnessseparableness",
    pos_rate=0.05,
)

# Get current date as string
date_str = datetime.now().strftime("%Y-%m-%d")

PROJECT_ROOT = Path(__file__).parent
GENERAL_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "outputs_for_publishing"
    / date_str
    / f"{best_run.wandb_group}"
    / f"{best_run.model}"
)
FIGURES_PATH = GENERAL_ARTIFACT_PATH / "figures"
TABLES_PATH = GENERAL_ARTIFACT_PATH / "tables"
ROBUSTNESS_PATH = FIGURES_PATH / "robustness"
