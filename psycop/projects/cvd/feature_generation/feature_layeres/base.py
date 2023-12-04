from collections.abc import Sequence
from typing import Protocol

from timeseriesflattener.feature_specs.single_specs import (
    AnySpec,
    OutcomeSpec,
    PredictorSpec,
    StaticSpec,
)

AnySpecType = AnySpec | PredictorSpec | OutcomeSpec | StaticSpec


class FeatureLayer(Protocol):
    def get_features(self, lookbehind_days: int) -> Sequence[PredictorSpec]:
        ...
