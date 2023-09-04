from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ModelCheckpoint:
    train_step: int
    model_state_dict: dict[Any, Any]
    optimizer_state_dict: dict[Any, Any]
    loss: float
    train_dataloader: DataLoader
    val_dataloader: DataLoader


class CheckpointSaver(Protocol):
    def __init__(self, checkpoint_path: Path, override_on_save: bool) -> None:
        ...

    def save(
        self,
        checkpoint: ModelCheckpoint,
    ):
        ...

    def load_latest(self) -> ModelCheckpoint | None:
        ...
