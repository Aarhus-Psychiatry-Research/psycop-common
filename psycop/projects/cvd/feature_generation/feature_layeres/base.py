from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
)
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


@dataclass(frozen=True)
class LayerNamedDataframe(NamedDataframe):
    layer: int

    @property
    def name(self) -> str:
        return f"{self.name}_layer_{self.layer}"
