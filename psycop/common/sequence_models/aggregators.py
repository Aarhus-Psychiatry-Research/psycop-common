from abc import abstractmethod

import torch
from torch import nn

from .registry import Registry


class Aggregator(nn.Module):
    @abstractmethod
    def forward(
        self,
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        pass


@Registry.layers.register("cls_aggregator")
class CLSAggregator(Aggregator):
    """
    Takes the hidden state corresponding to the first token (i.e. the CLS token).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        return last_hidden[:, 0, :]


@Registry.layers.register("average_pooler")
class AveragePooler(Aggregator):
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
