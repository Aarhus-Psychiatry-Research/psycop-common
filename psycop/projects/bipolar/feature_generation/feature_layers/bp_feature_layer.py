from collections.abc import Sequence
from typing import Protocol

from psycop.projects.bipolar.feature_generation.feature_layers.value_specification import (
    ValueSpecification,
)


class BpFeatureLayer(Protocol):
    def get_features(self, lookbehind_days: list[float]) -> Sequence[ValueSpecification]: ...
