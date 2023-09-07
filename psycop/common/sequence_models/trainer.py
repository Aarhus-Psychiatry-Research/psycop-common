"""
Defines the trainer class for sequence models
"""

import math
from typing import Protocol, Sequence

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import Self

from psycop.common.sequence_models.checkpoint_savers.base import (
    Checkpoint,
    CheckpointSaver,
    TrainingState,
)
from psycop.common.sequence_models.loggers.base import Logger

BatchWithLabels = tuple[dict[str, torch.Tensor], torch.Tensor]


# TODO: Super annoying module name, but what is the name of an optimizer + a model?
class PSYCOPModule(Protocol):
    optimizer: Optimizer

    def training_step(self, batch: BatchWithLabels) -> torch.Tensor:
        ...

    def validation_step(self, batch: BatchWithLabels) -> torch.Tensor:
        ...

    def get_state(self) -> TrainingState:
        ...

    def load_checkpoint(self, checkpoint: TrainingState):
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

        self.train_step = 0

        self.save_every_n_steps = save_every_n_steps
        self.checkpoint_savers = checkpoint_savers

    def fit(
        self,
        n_steps: int,
        model: PSYCOPModule,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        resume_from_latest_checkpoint: bool = True,
    ) -> Self:
        checkpoint_state = self._load_state_from_latest_checkpoint()

        if n_steps <= self.train_step:
            raise ValueError(
                f"Model is already trained to {self.train_step} steps, n_steps would result in no further training. Set a new n_steps > {self.train_step}."
            )

        if resume_from_latest_checkpoint and checkpoint_state is not None:
            print("Resuming from latest checkpoint")
            model.load_checkpoint(checkpoint=checkpoint_state.training_state)
            self.train_step = checkpoint_state.train_step
        else:
            if resume_from_latest_checkpoint and checkpoint_state is None:
                print("No checkpoint found, starting from scratch")
            else:
                print(
                    f"Resume from latest checkpoint is {resume_from_latest_checkpoint}, training model from scratch"
                )

        train_loss = []
        n_epochs = max(int(n_steps / len(train_dataloader)), 1)
        for _ in range(n_epochs):
            for batch in train_dataloader:
                loss = model.training_step(batch=batch)
                train_loss.append(loss)

                if self.train_step % self.validate_every_n_steps == 0:
                    self._evaluate(
                        model=model,
                        val_dataloader=val_dataloader,
                        train_loss=train_loss,
                        train_index=self.train_step,
                    )
                if self.train_step % self.save_every_n_steps == 0:
                    self._save_checkpoints(
                        model=model,
                        global_steps=self.train_step,
                        train_loss=train_loss,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                    )

                self.train_step += 1
                if self.train_step >= n_steps:
                    break

        return self

    def _save_checkpoints(
        self,
        model: PSYCOPModule,
        global_steps: int,
        train_loss: list[torch.Tensor],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ):
        train_loss_mean = float(torch.stack(train_loss).mean())

        for checkpointer in self.checkpoint_savers:
            checkpointer.save(
                Checkpoint(
                    run_name=self.logger.run_name,
                    train_step=global_steps,
                    loss=train_loss_mean,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    training_state=model.get_state(),
                )
            )

    def _evaluate(
        self,
        model: PSYCOPModule,
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

    def _load_state_from_latest_checkpoint(self) -> Checkpoint | None:
        """
        Loads the trainer from disk
        """
        for checkpointer in self.checkpoint_savers:
            checkpoint = checkpointer.load_latest()
            if checkpoint is None:
                break
            return checkpoint
        return None
