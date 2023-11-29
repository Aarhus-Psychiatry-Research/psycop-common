from collections.abc import Sequence
from typing import Protocol

from timeseriesflattener.feature_specs.single_specs import (
    AnySpec,
)


class SczBpFeatureLayer(Protocol):
    def get_features(self, lookbehind_days: list[float]) -> Sequence[AnySpec]:
        ...
