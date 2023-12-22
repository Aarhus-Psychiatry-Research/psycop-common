from collections.abc import Iterator
from dataclasses import dataclass

import torch


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
