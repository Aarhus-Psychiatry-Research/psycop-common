"""
Defines the trainer class for sequence models
"""

from dataclasses import dataclass
from tabnanny import check
from typing import Any, Protocol, Self, Sequence

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, dataloader

from psycop.common.sequence_models.checkpoint_savers.base import (
    CheckpointSaver,
)
from psycop.common.sequence_models.loggers.base import Logger


class TrainableModel(Protocol):
    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ...

    def validation_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ...

    def configure_optimizer(
        self, state_dict: dict[Any, Any] | None = None
    ) -> Optimizer:
        ...

    def state_dict(self) -> dict[str, float]:
        ...

    def load_state_dict(self, state_dict: dict[str, float]) -> Self:
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
        train_index: int,
        model: TrainableModel,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        resume_from_latest_checkpoint: bool = True,
    ) -> None:
        checkpoint_state = self._load_state_from_latest_checkpoint()

        if checkpoint_state is not None and resume_from_latest_checkpoint:
            model.load_state_dict(checkpoint_state.model_state_dict)
            optimizer = model.configure_optimizer(
                state_dict=checkpoint_state.optimizer_state_dict
            )
            train_index = checkpoint_state.n_steps
        elif resume_from_latest_checkpoint and checkpoint_state is None:
            print("No checkpoint found, starting from scratch")
            optimizer = model.configure_optimizer()
            train_index = 0

        # TODO: Figure out how to save the state of the dataloader sampler, so we can continue from the same point

        train_loss = []
        for batch in train_dataloader:
            loss = model.training_step(batch=batch)
            train_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if train_index % self.validate_every_n_steps == 0:
                self._evaluate(
                    model=model,
                    val_dataloader=val_dataloader,
                    train_loss=train_loss,
                    train_index=train_index,
                )

            if train_index % self.save_every_n_steps == 0:
                self._save_state(
                    model=model,
                    optimizer=optimizer,
                    global_steps=batch,
                    loss=float(loss),
                    train_dataloader=train_dataloader,
                )

            if train_index == train_index:
                break

            train_index += 1

    def _save_state(
        self,
        model: TrainableModel,
        global_steps: int,
        optimizer: Optimizer,
        loss: float,
        train_dataloader: DataLoader,
    ):
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()

        for checkpointer in self.checkpoint_savers:
            checkpointer.save(
                epoch=global_steps,
                model_state_dict=model_state,
                optimizer_state_dict=optimizer_state,
                loss=loss,
                dataloader=train_dataloader,
            )

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

    def _load_state_from_latest_checkpoint(self) -> ModelCheckpoint | None:
        """
        Loads the trainer from disk
        """
        for checkpointer in self.checkpoint_savers:
            checkpoint = checkpointer.load_latest()
            if checkpoint is None:
                break
            return checkpoint
        return None
