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
            self.embedding_module.d_model, embedding_module.vocab_size
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

        assert self.encoder_module.d_model == self.embedding_module.d_model

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

    def mask(self, patients: list[Patient]) -> tuple[list[Patient], torch.Tensor]:
        output_label = []
        output_token = []
        # For each patient, mask token with probability 15%
        for patient in patients:
            masked_lm_labels = [-1] * len(
                patient.events
            )  # -1 will be ignored in loss function
            for i, event in enumerate(patient.events):
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15
                    output_token.append(copy(event))
                    # 80% of the time, replace with [MASK]
                    if prob < 0.8:
                        patient.events[i] = self.embedding_module.diagnosis2idx[
                            "MASK"
                        ]  # TODO consider changing this to mask_token_idx
                    # 10% of the time, replace with random word
                    elif prob < 0.5:
                        patient.events[i] = random.randint(
                            0, self.embedding_module.vocab_size - 1
                        )
                    # -> rest 10% of the time, keep the original word
                    # append the original token to output (we will predict these later)
                    masked_lm_labels[i] = event
            output_label.append(masked_lm_labels)
        return output_token, output_label

    def masking_fn(self, patients: list[Patient]) -> dict[str, torch.Tensor]:
        """
        Masking function for the task
        """
        code, label = self.mask(patients)
        inputs = self.embedding_module.collate_fn(events, mask=masked_lm_labels)
        return inputs, masked_lm_labels
