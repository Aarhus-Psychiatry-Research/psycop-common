"""
Defines the trainer class for sequence models
"""

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        task: nn.Module,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
    ) -> None:
        self.task = task
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

    def train(self, steps: int) -> None:
        pass

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Performs a single training step
        """
        pass

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        pass

    def log(self, metrics: dict[str, float]) -> None:
        """
        Logs metrics to the logger
        """
        pass

    def save_to_disk(self, path: str) -> None:
        """
        Saves the trainer to disk including the optimizer state, the task state etc.
        """
        pass

    def load_from_disk(self, path: str) -> None:
        """
        Loads the trainer from disk
        """
        pass
