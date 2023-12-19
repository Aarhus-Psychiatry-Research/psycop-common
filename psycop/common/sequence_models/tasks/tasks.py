import logging
from collections.abc import Iterator, Sequence
from copy import copy
from dataclasses import dataclass
from typing import Literal

import lightning.pytorch as pl
import torch
from torch import nn
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC, MulticlassAUROC

from psycop.common.data_structures.patient import PatientSlice

from ..aggregators import Aggregator
from ..embedders.BEHRT_embedders import BEHRTEmbedder
from ..optimizers import LRSchedulerFn, OptimizerFn
from ..registry import Registry

log = logging.getLogger(__name__)


@dataclass
class BatchWithLabels:
    """
    A batch with labels.

    Attributes:
        inputs: A dictionary of padded sequence ids. Shape (batch_size, sequence_length).
        labels: A tensor of labels for the batch. Shape (batch_size, num_classes).
    """

    inputs: dict[str, torch.Tensor]
    labels: torch.Tensor

    def __iter__(self) -> Iterator[dict[str, torch.Tensor] | torch.Tensor]:
        return iter((self.inputs, self.labels))


@Registry.tasks.register("behrt")
class BEHRTForMaskedLM(pl.LightningModule):
    """An implementation of the BEHRT model for the masked language modeling task."""

    def __init__(
        self,
        embedder: BEHRTEmbedder,
        encoder: nn.Module,
        optimizer: OptimizerFn,
        lr_scheduler: LRSchedulerFn,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.embedder = embedder
        self.encoder = encoder
        self.optimizer = optimizer
        self.lr_scheduler_fn = lr_scheduler

        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.d_model = self.embedder.d_model

        if self.embedder.is_fitted:
            # Otherwise, the mlm_head will be initialized in on_fit_start
            self.mlm_head = nn.Linear(self.d_model, self.embedder.n_diagnosis_codes)

    def training_step(  # type: ignore
        self,
        batch: BatchWithLabels,
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        output = self.forward(batch.inputs, batch.labels)
        loss = output["loss"]
        # Update the weights
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: BatchWithLabels, batch_idx: int) -> torch.Tensor:  # type: ignore  # noqa: ARG002
        output = self.forward(inputs=batch.inputs, masked_lm_labels=batch.labels)
        self.log("val_loss", output["loss"])
        return output["loss"]

    def forward(  # type: ignore
        self,
        inputs: dict[str, torch.Tensor],
        masked_lm_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embedded_patients = self.embedder(inputs)
        encoded_patients = self.encoder(
            src=embedded_patients.src,
            src_key_padding_mask=embedded_patients.src_key_padding_mask,
        )

        logits = self.mlm_head(encoded_patients)
        masked_lm_loss = self.loss(
            logits.view(-1, logits.size(-1)),
            masked_lm_labels.view(-1),
        )  # (bs * seq_length, vocab_size), (bs * seq_length)
        return {"logits": logits, "loss": masked_lm_loss}

    @staticmethod
    def mask(
        diagnosis: torch.Tensor,
        n_diagnoses_in_vocab: int,
        mask_token_id: int,
        padding_mask: torch.Tensor,
        masking_prob: float = 0.15,
        replace_with_mask_prob: float = 0.8,
        replace_with_random_prob: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Masking function for the task
        """
        masked_labels = diagnosis.clone()
        # Mask 15 % of the tokens
        prob = torch.rand(diagnosis.shape)
        mask = prob < masking_prob
        masked_labels[~mask] = -1  # -1 will be ignored in loss function
        prob /= masking_prob
        # 80% of the time, replace with [MASK] token
        mask[mask.clone()] = prob[mask] < replace_with_mask_prob
        diagnosis[mask] = mask_token_id

        # 10% of the time, replace with random token
        prob /= 0.8
        mask[mask.clone()] = prob[mask] < replace_with_random_prob
        diagnosis[mask] = torch.randint(0, n_diagnoses_in_vocab - 1, mask.sum().shape)

        # Set padding to -1 to ignore in loss
        masked_labels[padding_mask] = -1
        # If no element in the batch is masked, mask the first element.
        # Is necessary to not get errors with small batch sizes, since the MLM module expects
        # at least one element to be masked.
        if torch.all(masked_labels == -1):
            masked_labels[0][0] = 1
        # -> rest 10% of the time, keep the original word
        return diagnosis, masked_labels

    def masking_fn(
        self,
        padded_sequence_ids: dict[str, torch.Tensor],
    ) -> BatchWithLabels:
        """
        Takes a dictionary of padded sequence ids and masks 15% of the tokens in the diagnosis sequence.
        """
        if not self.embedder.is_fitted:
            raise ValueError(
                "The embedder must be fitted before masking, so it knows which token_id is that mask. Call embedder.fit() before masking.",
            )

        self.mask_token_id = self.embedder.vocab.diagnosis["MASK"]
        padded_sequence_ids = copy(padded_sequence_ids)
        padding_mask = padded_sequence_ids["is_padding"] == 1
        # Perform masking
        masked_sequence, masked_labels = self.mask(
            diagnosis=padded_sequence_ids["diagnosis"],
            n_diagnoses_in_vocab=self.embedder.n_diagnosis_codes,
            mask_token_id=self.mask_token_id,
            padding_mask=padding_mask,
        )
        # Replace padded_sequence_ids with masked_sequence
        padded_sequence_ids["diagnosis"] = masked_sequence
        return BatchWithLabels(padded_sequence_ids, masked_labels)

    def collate_fn(
        self,
        patient_slices: list[PatientSlice],
    ) -> BatchWithLabels:
        """
        Takes a list of PredictionTime and returns a dictionary of padded sequence ids.
        """
        padded_sequence_ids = self.embedder.collate_patient_slices(
            patient_slices,
        )
        # Masking
        batch_with_labels = self.masking_fn(padded_sequence_ids)
        return batch_with_labels

    def on_fit_start(self) -> None:
        """Pytorch lightning hook. Called at the beginning of every fit."""
        self.mlm_head = nn.Linear(self.d_model, self.embedder.n_diagnosis_codes)

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
        slices_with_content = [
            patient_slice
            for patient_slice in reformatted_slices
            if patient_slice.temporal_events
        ]

        if len(reformatted_slices) != len(slices_with_content):
            n_filtered = len(reformatted_slices) - len(slices_with_content)
            log.warning(
                f"Lost {n_filtered} patients ({round(n_filtered / len(patient_slices) * 100, 2)}%) after filtering and mapping diagnosis codes.",
            )

        return slices_with_content


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
        patient_slices_with_labels: list[tuple[PatientSlice, int]],
    ) -> BatchWithLabels:
        """
        Takes a list of patients and returns a dictionary of padded sequence ids.
        """
        patient_slices, outcomes = list(zip(*patient_slices_with_labels))  # type: ignore
        patient_slices: list[PatientSlice] = list(patient_slices)
        padded_sequence_ids = self.embedder.collate_patient_slices(
            patient_slices,
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
