import os
from pathlib import Path
from typing import Any

import torch

from psycop.common.sequence_models.checkpoint_savers.base import CheckpointSaver
from psycop.common.sequence_models.trainer import Trainer


class CheckpointToDisk(CheckpointSaver):
    def __init__(self, checkpoint_path: Path, override_on_save: bool) -> None:
        self.checkpoint_path = checkpoint_path
        self.override_on_save = override_on_save

    def save(
        self,
        epoch: int,
        model_state_dict: dict[Any, Any],
        optimizer_state_dict: dict[Any, Any],
        loss: float,
        run_name: str,
    ) -> None:
        torch.save(
            obj={
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "loss": loss,
            },
            f=self.checkpoint_path / f"{run_name}.pkt",
        )

    def load_latest(self) -> Trainer:
        # Get file with the latest modified date in self.checkpoint
        files = self.checkpoint_path.glob(r".+\.pkt")
        latest_file = max(files, key=os.path.getctime)

        saved_info = torch.load(latest_file)

        # Load checkpoint
