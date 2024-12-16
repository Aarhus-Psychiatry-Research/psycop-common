from collections.abc import Sequence
from typing import Protocol

from psycop.projects.scz_bp.feature_generation.feature_layers.value_specification import (
    ValueSpecification,
)


class ClozapineFeatureLayer(Protocol):
    def get_features(self, lookbehind_days: list[float]) -> Sequence[ValueSpecification]: ...
