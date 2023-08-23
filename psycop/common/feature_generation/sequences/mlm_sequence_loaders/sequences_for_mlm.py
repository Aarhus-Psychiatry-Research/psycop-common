from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from psycop.common.feature_generation.sequences.mlm_sequence_loaders.diagnoses_sequences import (
        EventDfLoader,
    )
    from psycop.common.feature_generation.sequences.timeseries_windower_python.patient import (
        Patient,
    )


class AbstractMLMDataLoader(ABC):
    @staticmethod
    @abstractmethod
    def get_train_set(loaders: Sequence[EventDfLoader]) -> list[Patient]:
        ...
