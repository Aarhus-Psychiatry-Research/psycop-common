from collections.abc import Sequence
from typing import Protocol

from timeseriesflattener.v1.aggregation_fns import AggregationFunType
from timeseriesflattener.v1.feature_specs.single_specs import (
    AnySpec,
    OutcomeSpec,
    PredictorSpec,
    StaticSpec,
)

AnySpecType = AnySpec | PredictorSpec | OutcomeSpec | StaticSpec


class FeatureLayer(Protocol):
    def get_features(
        self, lookbehind_days: Sequence[int], aggregation_fns: Sequence[AggregationFunType]
    ) -> Sequence[PredictorSpec]:
        ...
