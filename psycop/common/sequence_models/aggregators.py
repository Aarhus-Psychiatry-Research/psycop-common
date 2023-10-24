from abc import abstractmethod

import torch
from torch import nn


class AggregationModule(nn.Module):
    @abstractmethod
    def forward(
        self,
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        pass


class CLSAggregationModule(AggregationModule):
    """
    Takes the hidden state corresponding to the first token (i.e. the CLS token).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return last_hidden[:, 0, :]


class AveragePooler(AggregationModule):
    """
    Parameter-free poolers to get the sentence embedding
    derived from https://github.com/princeton-nlp/SimCSE/blob/13361d0e29da1691e313a94f003e2ed1cfa97fef/simcse/models.py#LL49C1-L84C1
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            -1,
        ).unsqueeze(-1)
