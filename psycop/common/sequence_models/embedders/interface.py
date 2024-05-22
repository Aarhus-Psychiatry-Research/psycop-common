from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

from psycop.common.data_structures.patient import PatientSlice


@dataclass(frozen=True)
class EmbeddedSequence:
    """
    A dataclass containing an embedded sequence and a mask indicating which tokens are padding tokens

    Attributes:
        src: A tensor containing the embedded token sequence. Shape (batch, sequence length, d_model)
        src_key_padding_mask: A tensor containing boolean values indicating which tokens are padding tokens
            (True) and which are not (False). Shape: (batch, sequence length)
    """

    src: torch.Tensor
    src_key_padding_mask: torch.Tensor


@runtime_checkable
class PatientSliceEmbedder(Protocol):
    """
    Interface for embedding modules
    """

    is_fitted: bool
    d_model: int

    def __init__(self, *args: Any) -> None: ...

    def forward(self, inputs: dict[str, torch.Tensor]) -> EmbeddedSequence: ...

    def collate_patient_slices(
        self, patient_slices: Sequence[PatientSlice]
    ) -> dict[str, torch.Tensor]: ...

    def fit(self, patient_slices: Sequence[PatientSlice]) -> None: ...

    def reformat(self, patient_slices: Sequence[PatientSlice]) -> list[PatientSlice]: ...
