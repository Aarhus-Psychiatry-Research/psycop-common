import os
from pathlib import Path

import torch

from psycop.common.sequence_models.checkpoint_savers.base import (
    Checkpoint,
    CheckpointSaver,
)


class CheckpointToDisk(CheckpointSaver):
    def __init__(self, checkpoint_path: Path, override_on_save: bool) -> None:
        self.checkpoint_path = checkpoint_path
        self.override_on_save = override_on_save

    def save(
        self,
        checkpoint: Checkpoint,
    ) -> None:
        if self.override_on_save:
            file_name = f"{checkpoint.run_name}.pt"
        else:
            file_name = f"{checkpoint.run_name}_step_{checkpoint.train_step}.pt"

        torch.save(
            obj=checkpoint,
            f=self.checkpoint_path / file_name,
        )

    def load_latest(self) -> Checkpoint | None:
        # Get file with the latest modified date in self.checkpoint
        files = list(self.checkpoint_path.glob(r"*.pt"))
        if len(files) == 0:
            return None

        latest_file = max(files, key=os.path.getctime)
        return torch.load(latest_file)
