"""
Defines the trainer class for sequence models
"""

from pathlib import Path
from typing import Protocol, Sequence

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from psycop.common.sequence_models.loggers.base import Logger


class TrainableModel(Protocol):
    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ...

    def validation_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ...

    def configure_optimizer(self) -> Optimizer:
        ...


class CheckpointSaver(Protocol):
    def __init__(self, checkpoint_path: Path, override_on_save: bool) -> None:
        ...

    def save(self) -> None:
        ...

    def load_latest(self) -> None:
        ...


class Trainer:
    def __init__(
        self,
        device: torch.device,
        validate_every_n_steps: int,
        n_samples_to_validate_on: int,
        logger: Logger,
        checkpoint_savers: Sequence[CheckpointSaver],
        save_every_n_steps: int,
    ) -> None:
        self.device = device

        self.validate_every_n_steps = validate_every_n_steps
        self.n_samples_to_validate_on = n_samples_to_validate_on
        self.logger = logger

        self.save_every_n_steps = save_every_n_steps
        self.checkpoint_savers = checkpoint_savers

    def fit(
        self,
        n_steps: int,
        model: TrainableModel,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> None:
        optimizer = model.configure_optimizer()

        train_loss = []
        for train_index, batch in enumerate(train_dataloader):
            loss = model.training_step(batch=batch)
            train_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if n_steps % self.validate_every_n_steps == 0:
                self._evaluate(
                    model=model,
                    val_dataloader=val_dataloader,
                    train_loss=train_loss,
                    train_index=train_index,
                )

            if n_steps % self.save_every_n_steps == 0:
                for checkpointer in self.checkpoint_savers:
                    checkpointer.save()

            if n_steps == train_index:
                break

    def _evaluate(
        self,
        model: TrainableModel,
        val_dataloader: DataLoader,
        train_loss: list[torch.Tensor],
        train_index: int,
    ):
        val_loss: list[torch.Tensor] = []
        for val_index, val_batch in enumerate(val_dataloader):
            if val_index == self.n_samples_to_validate_on:
                break
            val_loss.append(model.validation_step(batch=val_batch))

        val_loss_mean = float(torch.stack(val_loss).mean())

        train_loss_mean = float(torch.stack(train_loss).mean())
        train_loss = []

        self.logger.log_metrics(
            metrics={
                "Training loss": train_loss_mean,
                "Validation loss": val_loss_mean,
                "Training step": train_index,
            }
        )

    def resume_training_from_latest_checkpoint(self) -> None:
        """
        Loads the trainer from disk
        """
        # TODO - Should this be in init or .fit so we can automatically resume by just re-running a script?
        for checkpointer in self.checkpoint_savers:
            result = checkpointer.load_latest()
            if result is not None:
                self = result
                break
