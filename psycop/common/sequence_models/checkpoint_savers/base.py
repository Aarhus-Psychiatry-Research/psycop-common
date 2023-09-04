from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ModelCheckpoint:
    n_steps: int
    model_state_dict: dict[Any, Any]
    optimizer_state_dict: dict[Any, Any]
    loss: float


class CheckpointSaver(Protocol):
    def __init__(self, checkpoint_path: Path, override_on_save: bool) -> None:
        ...

    def save(
        self,
        epoch: int,
        model_state_dict,
        optimizer_state_dict,
        loss: float,
        dataloader: DataLoader,
    ):
        ...

    def load_latest(self) -> ModelCheckpoint | None:
        ...
