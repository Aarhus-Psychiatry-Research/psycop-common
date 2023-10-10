from typing import Any, Protocol

import torch

from psycop.common.data_structures import Patient


class Embedder(Protocol):
    """
    Interface for embedding modules
    """

    def __init__(self, *args: Any) -> None:
        ...

    def __call__(self, *args: Any) -> torch.Tensor:
        ...

    def forward(self, *args: Any) -> torch.Tensor:
        ...

    def collate_patients(self, patients: list[Patient]) -> dict[str, torch.Tensor]:
        ...

    def fit(self, patients: list[Patient], *args: Any) -> None:
        ...
