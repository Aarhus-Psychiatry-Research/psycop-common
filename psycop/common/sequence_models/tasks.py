import random
from copy import copy

import torch
from torch import nn

from psycop.common.data_structures import Patient


class BEHRTMaskingTask(nn.Module):
    """Masked Language Model (MLM) task for BEHRT implementation"""

    def __init__(
        self,
        embedding_module: nn.Module,
        encoder_module: nn.Module,
    ):
        super().__init__()

        self.embedding_module = embedding_module
        self.encoder_module = encoder_module
        self.mlm_head = nn.Linear(
            self.embedding_module.d_model, self.embedding_module.n_diagnosis_codes
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        masked_lm_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embedded_patients = self.embedding_module(**inputs)
        encoded_patients = self.encoder_module(embedded_patients)
        logits = self.mlm_head(encoded_patients)

        masked_lm_loss = self.loss(
            logits.view(-1, logits.size(-1)), masked_lm_labels.view(-1)
        )  # (bs * seq_length, vocab_size), (bs * seq_length)
        return {"logits": logits, "masked_lm_loss": masked_lm_loss}

    def mask(self, diagnosis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Masking function for the task
        """
        masked_lm_labels = diagnosis.clone()

        # Mask 15 % of the tokens
        prob = torch.rand(diagnosis.shape)
        mask = prob < 0.15

        masked_lm_labels[~mask] = -1  # -1 will be ignored in loss function

        prob /= 0.15

        # 80% of the time, replace with [MASK] token
        mask[mask.clone()] = prob[mask] < 0.8
        diagnosis[mask] = self.embedding_module.mask_token_id  # TODO

        # 10% of the time, replace with random token
        prob /= 0.8
        mask[mask.clone()] = prob[mask] < 0.1
        diagnosis[mask] = torch.randint(
            0, self.embedding_module.n_diagnosis_codes - 1, mask.sum().shape
        )  # TODO

        # -> rest 10% of the time, keep the original word

        return diagnosis, masked_lm_labels

    def masking_fn(
        self, padded_sequence_ids: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Takes a dictionary of padded sequence ids and masks 15% of the tokens in the diagnosis sequence.
        """
        padded_sequence_ids = copy(padded_sequence_ids)

        # Perform masking
        masked_sequence, masked_labels = self.mask(padded_sequence_ids["diagnosis"])

        # Set padding to -1 to ignore in loss
        masked_labels[padded_sequence_ids["is_padding"] == 1] = -1

        # Replace padded_sequence_ids with masked_sequence
        padded_sequence_ids["diagnosis"] = masked_sequence

        return padded_sequence_ids, masked_labels
