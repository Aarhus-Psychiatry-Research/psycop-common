import logging
from collections.abc import Sequence
from copy import copy
from typing import final

import torch
from torch import nn

from psycop.common.data_structures.patient import PatientSlice
from psycop.common.sequence_models.datatypes import BatchWithLabels

from ..embedders.BEHRT_embedders import BEHRTEmbedder
from ..optimizers import LRSchedulerFn, OptimizerFn
from ..registry import Registry
from .base_pretraining_task import BasePatientSlicePretrainer, Metrics

log = logging.getLogger(__name__)


@Registry.tasks.register("behrt")
@final  # This is not an ABC, must not contain abstract methods
class BEHRTForMaskedLM(BasePatientSlicePretrainer):
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
        self._embedder = embedder
        self._encoder = encoder
        self.optimizer = optimizer
        self.lr_scheduler_fn = lr_scheduler

        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.d_model = self._embedder.d_model

        if self._embedder.is_fitted:
            # Otherwise, the mlm_head will be initialized in on_fit_start
            self._initialise_mlm_head()

    def _initialise_mlm_head(self):
        self.mlm_head = nn.Linear(self.d_model, self._embedder.n_diagnosis_codes)

    def training_step(
        self,
        batch: BatchWithLabels,
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        output = self.forward(batch.inputs, batch.labels)
        loss = output["loss"]
        # Update the weights
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self,
        batch: BatchWithLabels,
        batch_idx: int,
    ) -> torch.Tensor:  # noqa: ARG002
        output = self.forward(inputs=batch.inputs, labels=batch.labels)
        self.log("val_loss", output["loss"])
        return output["loss"]

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Metrics:
        embedded_patients = self._embedder(inputs)
        encoded_patients = self._encoder(
            src=embedded_patients.src,
            src_key_padding_mask=embedded_patients.src_key_padding_mask,
        )

        logits = self.mlm_head(encoded_patients)
        masked_lm_loss = self.loss(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
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
        if not self._embedder.is_fitted:
            raise ValueError(
                "The embedder must know which token_id corresponds to the mask. Therefore, the embedder must be fitted before masking. Call embedder.fit() before masking.",
            )

        self.mask_token_id = self._embedder.vocab.diagnosis["MASK"]
        padded_sequence_ids = copy(padded_sequence_ids)
        padding_mask = padded_sequence_ids["is_padding"] == 1
        # Perform masking
        masked_sequence, masked_labels = self.mask(
            diagnosis=padded_sequence_ids["diagnosis"],
            n_diagnoses_in_vocab=self._embedder.n_diagnosis_codes,
            mask_token_id=self.mask_token_id,
            padding_mask=padding_mask,
        )
        # Replace padded_sequence_ids with masked_sequence
        padded_sequence_ids["diagnosis"] = masked_sequence
        return BatchWithLabels(padded_sequence_ids, masked_labels)

    def collate_fn(
        self,
        patient_slices: Sequence[PatientSlice],
    ) -> BatchWithLabels:
        """
        Takes a list of PredictionTime and returns a dictionary of padded sequence ids.
        """
        padded_sequence_ids = self._embedder.collate_patient_slices(
            patient_slices,
        )
        # Masking
        batch_with_labels = self.masking_fn(padded_sequence_ids)
        return batch_with_labels

    def on_fit_start(self) -> None:
        """Pytorch lightning hook. Called at the beginning of every fit."""
        self._initialise_mlm_head()

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer],
        list[torch.optim.lr_scheduler._LRScheduler],  # type: ignore
    ]:
        optimizer = self.optimizer(self.parameters())
        lr_scheduler = self.lr_scheduler_fn(optimizer)
        return [optimizer], [lr_scheduler]

    def filter_and_reformat(
        self,
        patient_slices: Sequence[PatientSlice],
    ) -> Sequence[PatientSlice]:
        reformatted_slices = self._embedder.reformat(patient_slices)
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

    @property
    def embedder(self) -> BEHRTEmbedder:
        return self._embedder

    @property
    def encoder(self) -> nn.Module:
        return self._encoder
