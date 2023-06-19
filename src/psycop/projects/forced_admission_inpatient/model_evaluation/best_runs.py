from dataclasses import dataclass


@dataclass
class BestRun:
    wandb_group: str
    model: str
    pos_rate: float


best_run = BestRun(
    wandb_group="bonnetiere-coarrange",
    model="impertriturature",
    pos_rate=0.05,
)
