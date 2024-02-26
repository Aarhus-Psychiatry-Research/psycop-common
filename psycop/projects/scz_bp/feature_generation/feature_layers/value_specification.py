from typing import Union

from timeseriesflattener import (
    BooleanOutcomeSpec,
    OutcomeSpec,
    PredictorSpec,
    StaticSpec,
    TimeDeltaSpec,
)

ValueSpecification = Union[
    PredictorSpec, OutcomeSpec, BooleanOutcomeSpec, TimeDeltaSpec, StaticSpec
]
