from collections.abc import Sequence
from typing import Literal, final

import torch
from torch import nn
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC, MulticlassAUROC

from psycop.common.sequence_models.tasks.patientslice_classifier_base import Logits, Loss

from ...data_structures.patient import PatientSlice
from ...data_structures.prediction_time import PredictionTime
from ..aggregators import Aggregator
from ..datatypes import BatchWithLabels
from ..embedders.interface import PatientSliceEmbedder
from ..optimizers import LRSchedulerFn, OptimizerFn
from ..registry import SequenceRegistry
from .patientslice_classifier_base import BasePredictionTimeClassifier, PredictedProbabilities


@SequenceRegistry.tasks.register("patient_slice_classifier")
@final  # This is not an ABC, must not contain abstract methods
class PatientSliceClassifier(BasePredictionTimeClassifier):
    def __init__(
        self,
        embedder: PatientSliceEmbedder,
        encoder: nn.Module,
        aggregator: Aggregator,
        optimizer: OptimizerFn,
        lr_scheduler: LRSchedulerFn,
        num_classes: int = 2,
    ):
        super().__init__()
        self._embedder = embedder
        self._encoder = encoder
        self.optimizer = optimizer
        self.lr_scheduler_fn = lr_scheduler
        self._aggregator = aggregator

        self.d_model: int = embedder.d_model

        self.is_binary = num_classes == 2
        self.num_classes = num_classes
        if self.is_binary:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        if self.is_binary:
            self.classification_head = nn.Linear(self.d_model, 1)
        else:
            self.classification_head = nn.Linear(self.d_model, num_classes)

        self.metric_fns = self.create_metrics(num_classes)

    @staticmethod
    def create_metrics(num_classes: int) -> dict[str, Metric]:
        is_binary = num_classes == 2
        if is_binary:
            return {"AUROC": BinaryAUROC()}
        return {"AUROC (macro)": MulticlassAUROC(num_classes=num_classes)}

    def training_step(
        self,
        batch: BatchWithLabels,
        batch_idx: int,  # noqa: ARG002
    ) -> Loss:
        loss = self._step(batch=batch, mode="Training")
        return loss

    def validation_step(
        self,
        batch: BatchWithLabels,
        batch_idx: int,  # noqa: ARG002
    ) -> Loss:
        loss = self._step(batch=batch, mode="Validation")
        return loss

    def _step(self, batch: BatchWithLabels, mode: Literal["Validation", "Training"]) -> Loss:
        """
        Logs the metrics for the given mode.
        """
        logits = self.forward(batch.inputs)
        metrics = self.calculate_metrics(logits, batch.labels)

        loss = self._calculate_loss(batch.labels, logits)
        metrics["loss"] = loss

        for metric_name, metric in metrics.items():
            self.log(f"{mode} {metric_name}", metric)

        return loss

    def forward(self, inputs: dict[str, torch.Tensor]) -> Logits:
        embedded_patients = self.embedder.forward(inputs)
        encoded_patients = self._encoder.forward(
            src=embedded_patients.src, src_key_padding_mask=embedded_patients.src_key_padding_mask
        )

        # Aggregate the sequence
        is_padding = embedded_patients.src_key_padding_mask
        aggregated_patients = self._aggregator(encoded_patients, attention_mask=~is_padding)

        # Classification head
        return self.classification_head(aggregated_patients)

    def _calculate_loss(self, labels: torch.Tensor, logits: Logits) -> Loss:
        if self.is_binary:
            _labels = labels.unsqueeze(-1).float()
        else:
            # If not binary convert to one-hot encoding
            _labels = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()

        loss = self.loss(logits, _labels)
        return loss

    def calculate_metrics(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Calculates the metrics for the task.
        """
        # Calculate the metrics
        metrics = {}
        probs = self._logits_to_probabilities(logits)

        for metric_name, metric in self.metric_fns.items():
            metrics[metric_name] = metric(probs, labels)

        return metrics

    def _logits_to_probabilities(self, logits: Logits) -> PredictedProbabilities:
        if self.is_binary:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=1)
        return probs

    def collate_fn(self, prediction_times: Sequence[PredictionTime]) -> BatchWithLabels:
        """
        Takes a list of patients and returns a dictionary of padded sequence ids.
        """
        patient_slices, outcomes = list(
            zip(*[(p.patient_slice, int(p.outcome)) for p in prediction_times])
        )
        padded_sequence_ids = self.embedder.collate_patient_slices(list(patient_slices))

        outcome_tensor = torch.tensor(outcomes)
        return BatchWithLabels(inputs=padded_sequence_ids, labels=outcome_tensor)

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer], list[torch.optim.lr_scheduler._LRScheduler]  # type: ignore
    ]:
        optimizer = self.optimizer(self.parameters())
        lr_scheduler = self.lr_scheduler_fn(optimizer)
        return [optimizer], [lr_scheduler]

    def filter_and_reformat(self, patient_slices: Sequence[PatientSlice]) -> Sequence[PatientSlice]:
        reformatted_slices = self.embedder.reformat(patient_slices)
        return reformatted_slices

    def predict_step(
        self,
        batch: BatchWithLabels,
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> PredictedProbabilities:
        logits = self.forward(batch.inputs)
        return self._logits_to_probabilities(logits)

    @property
    def embedder(self) -> PatientSliceEmbedder:
        return self._embedder

    @property
    def aggregator(self) -> Aggregator:
        return self._aggregator

    @property
    def encoder(self) -> nn.Module:
        return self._encoder
