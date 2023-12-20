from abc import ABC, abstractmethod
from collections.abc import Sequence

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import (
    OptimizerLRScheduler,
)
from torch import nn

from ...data_structures.patient import PatientSlice
from ...data_structures.prediction_time import PredictionTime
from ..aggregators import Aggregator
from ..datatypes import BatchWithLabels
from ..embedders.interface import PatientSliceEmbedder

Logits = torch.Tensor
Loss = torch.Tensor
PredictedProbabilities = torch.Tensor


class BasePatientSliceClassifier(ABC, pl.LightningModule):
    @abstractmethod
    def collate_fn(
        self,
        prediction_times: Sequence[PredictionTime],
    ) -> BatchWithLabels:
        ...

    @abstractmethod
    def training_step(self, batch: BatchWithLabels, batch_idx: int) -> torch.Tensor:  # type: ignore
        ...

    @abstractmethod
    def validation_step(self, batch: BatchWithLabels, batch_idx: int) -> torch.Tensor:  # type: ignore
        ...

    @abstractmethod
    def forward(  # type: ignore
        self,
        inputs: dict[str, torch.Tensor],
    ) -> Logits:
        ...

    @abstractmethod
    def filter_and_reformat(
        self,
        patient_slices: Sequence[PatientSlice],
    ) -> Sequence[PatientSlice]:
        ...

    def configure_optimizers(
        self,
    ) -> OptimizerLRScheduler:
        ...

    @abstractmethod
    def predict_step(  # type: ignore
        self,
        batch: BatchWithLabels,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> PredictedProbabilities:
        ...

    @property
    @abstractmethod
    def embedder(self) -> PatientSliceEmbedder:
        ...

    @property
    @abstractmethod
    def aggregator(self) -> Aggregator:
        ...

    @property
    @abstractmethod
    def encoder(self) -> nn.Module:
        ...
