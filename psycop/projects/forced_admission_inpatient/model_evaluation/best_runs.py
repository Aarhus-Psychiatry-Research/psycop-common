from dataclasses import dataclass


@dataclass
class BestRun:
    wandb_group: str
    model: str
    pos_rate: float


best_run = BestRun(
    wandb_group="physiurgic-letterleaf",
    model="basidiosporeneathmost ",
    pos_rate=0.05,
)
