import torch
from torch import nn


class BEHRTMaskingTask(nn.Module):
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
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)  # TODO check this

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
        )  # TODO check this
        return {"logits": logits, "masked_lm_loss": masked_lm_loss}

    def mask(self, patients: list[Patient]) -> tuple[list[Patient], torch.Tensor]:
        

    def masking_fn(self, patients: list[Patient]) -> dict[str, torch.Tensor]:
        """
        Masking function for the task
        """
        masked_lm_labels = self.mask(events)
        inputs = self.embedding_module.collate_fn(events, mask=masked_lm_labels)
        return inputs, masked_lm_labels
