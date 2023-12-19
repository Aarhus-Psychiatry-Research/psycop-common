from collections.abc import Sequence
from typing import Literal

import lightning.pytorch as pl
import torch
from torch import nn
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC, MulticlassAUROC

from ...data_structures.patient import PatientSlice
from ...data_structures.prediction_time import PredictionTime
from ..aggregators import Aggregator
from ..datatypes import BatchWithLabels
from ..embedders.BEHRT_embedders import BEHRTEmbedder
from ..optimizers import LRSchedulerFn, OptimizerFn
from ..registry import Registry


@Registry.tasks.register("clf_encoder")
class EncoderForClassification(pl.LightningModule):
    """
    A BEHRT model for the classification task.
    """

    def __init__(
        self,
        embedder: BEHRTEmbedder,
        encoder: nn.Module,
        aggregation_module: Aggregator,
        optimizer: OptimizerFn,
        lr_scheduler: LRSchedulerFn,
        num_classes: int = 2,
    ):
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.optimizer = optimizer
        self.lr_scheduler_fn = lr_scheduler
        self.aggregation_module = aggregation_module

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

    def training_step(self, batch: BatchWithLabels, batch_idx: int) -> torch.Tensor:  # type: ignore # noqa: ARG002
        output = self.forward(batch.inputs, batch.labels)
        loss = output["loss"]
        self.log_step("Training", output)
        return loss

    def validation_step(self, batch: BatchWithLabels, batch_idx: int) -> torch.Tensor:  # type: ignore  # noqa: ARG002
        output = self.forward(inputs=batch.inputs, labels=batch.labels)
        self.log_step("Validation", output)
        return output["loss"]

    def log_step(
        self,
        mode: Literal["Validation", "Training"],
        output: dict[str, torch.Tensor],
    ) -> None:
        """
        Logs the metrics for the given mode.
        """
        for metric_name, metric in output.items():
            self.log(f"{mode} {metric_name}", metric)

    def forward(  # type: ignore
        self,
        inputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embedded_patients = self.embedder(inputs)
        encoded_patients = self.encoder(
            src=embedded_patients.src,
            src_key_padding_mask=embedded_patients.src_key_padding_mask,
        )

        # Aggregate the sequence
        is_padding = embedded_patients.src_key_padding_mask
        aggregated_patients = self.aggregation_module(
            encoded_patients,
            attention_mask=~is_padding,
        )

        # Classification head
        logits = self.classification_head(aggregated_patients)
        if self.is_binary:
            _labels = labels.unsqueeze(-1).float()
        else:
            # If not binary convert to one-hot encoding
            _labels = torch.nn.functional.one_hot(
                labels,
                num_classes=self.num_classes,
            ).float()
        loss = self.loss(logits, _labels)  # type: ignore

        metrics = self.calculate_metrics(logits, labels)

        return {"loss": loss, **metrics}

    def calculate_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Calculates the metrics for the task.
        """
        # Calculate the metrics
        metrics = {}
        if self.is_binary:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=1)

        for metric_name, metric in self.metric_fns.items():
            metrics[metric_name] = metric(probs, labels)

        return metrics

    def collate_fn(
        self,
        prediction_times: Sequence[PredictionTime],
    ) -> BatchWithLabels:
        """
        Takes a list of patients and returns a dictionary of padded sequence ids.
        """
        patient_slices, outcomes = list(
            zip(*[(p.patient_slice, int(p.outcome)) for p in prediction_times]),
        )
        padded_sequence_ids = self.embedder.collate_patient_slices(
            list(patient_slices),
        )

        outcome_tensor = torch.tensor(outcomes)
        return BatchWithLabels(inputs=padded_sequence_ids, labels=outcome_tensor)

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer],
        list[torch.optim.lr_scheduler._LRScheduler],  # type: ignore
    ]:  # type: ignore
        optimizer = self.optimizer(self.parameters())
        lr_scheduler = self.lr_scheduler_fn(optimizer)
        return [optimizer], [lr_scheduler]

    def filter_and_reformat(
        self,
        patient_slices: Sequence[PatientSlice],
    ) -> Sequence[PatientSlice]:
        reformatted_slices = self.embedder.reformat(patient_slices)
        return reformatted_slices
