from copy import copy
from typing import Any

import lightning.pytorch as pl
import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup

from psycop.common.data_structures.patient import Patient

from .embedders.BEHRT_embedders import BEHRTEmbedder

BatchWithLabels = tuple[dict[str, torch.Tensor], torch.Tensor]


class BEHRTForMaskedLM(pl.LightningModule):
    """An implementation of the BEHRT model for the masked language modeling task."""

    def __init__(
        self,
        embedding_module: BEHRTEmbedder,
        encoder_module: nn.Module,
        optimizer_kwargs: dict[str, Any],
        lr_scheduler_kwargs: dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.embedding_module = embedding_module
        self.encoder_module = encoder_module
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = lr_scheduler_kwargs

        self.d_model = self.embedding_module.d_model
        self.mask_token_id = self.embedding_module.vocab.diagnosis["MASK"]

        self.mlm_head = nn.Linear(self.d_model, self.embedding_module.n_diagnosis_codes)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def training_step(  # type: ignore
        self,
        batch: BatchWithLabels,
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        output = self.forward(batch[0], batch[1])
        loss = output["loss"]
        # Update the weights
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: BatchWithLabels, batch_idx: int) -> torch.Tensor:  # type: ignore  # noqa: ARG002
        output = self.forward(inputs=batch[0], masked_lm_labels=batch[1])
        self.log("val_loss", output["loss"])
        return output["loss"]

    def forward(  # type: ignore
        self,
        inputs: dict[str, torch.Tensor],
        masked_lm_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embedded_patients = self.embedding_module(inputs)
        encoded_patients = self.encoder_module(embedded_patients)
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
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Takes a dictionary of padded sequence ids and masks 15% of the tokens in the diagnosis sequence.
        """
        padded_sequence_ids = copy(padded_sequence_ids)
        padding_mask = padded_sequence_ids["is_padding"] == 1
        # Perform masking
        masked_sequence, masked_labels = self.mask(
            diagnosis=padded_sequence_ids["diagnosis"],
            n_diagnoses_in_vocab=self.embedding_module.n_diagnosis_codes,
            mask_token_id=self.mask_token_id,
            padding_mask=padding_mask,
        )
        # Replace padded_sequence_ids with masked_sequence
        padded_sequence_ids["diagnosis"] = masked_sequence
        return padded_sequence_ids, masked_labels

    def collate_fn(
        self,
        patients: list[Patient],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Takes a list of patients and returns a dictionary of padded sequence ids.
        """
        padded_sequence_ids = self.embedding_module.collate_patients(patients)
        # Masking
        padded_sequence_ids, masked_labels = self.masking_fn(padded_sequence_ids)
        return padded_sequence_ids, masked_labels

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LambdaLR]]:
        optimizer = AdamW(
            self.parameters(),
            correct_bias=False,
            **self.optimizer_kwargs,
        )

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            **self.scheduler_kwargs,
        )
        return [optimizer], [lr_scheduler]
