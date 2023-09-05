from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainingState:
    model_state_dict: dict[Any, Any]
    optimizer_state_dict: dict[Any, Any]


@dataclass(frozen=True)
class Checkpoint:
    run_name: str
    train_step: int
    loss: float
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    training_state: TrainingState


class CheckpointSaver(Protocol):
    def __init__(self, checkpoint_path: Path, override_on_save: bool) -> None:
        ...

    def save(
        self,
        checkpoint: Checkpoint,
    ):
        ...

    def load_latest(self) -> Checkpoint | None:
        ...
