"""Feature specification module."""

import datetime as dt
import logging
from typing import TYPE_CHECKING

from timeseriesflattener import OutcomeSpec, ValueFrame
from timeseriesflattener.aggregators import MaxAggregator

from psycop.projects.bipolar.cohort_definition.diagnosis_timestamps.first_bipolar_diagnosis import (
    get_first_bipolar_diagnosis,
)
from psycop.projects.bipolar.feature_generation.feature_layers.bp_layer_1 import (
    BpLayer1,
)
from psycop.projects.bipolar.feature_generation.feature_layers.bp_layer_2 import (
    BpLayer2,
)
from psycop.projects.bipolar.feature_generation.feature_layers.bp_layer_3 import (
    BpLayer3,
)
from psycop.projects.bipolar.feature_generation.feature_layers.bp_layer_4 import (
    BpLayer4,
)
from psycop.projects.bipolar.feature_generation.feature_layers.value_specification import (
    ValueSpecification,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger(__name__)


BpFeatureLayers = {1: BpLayer1, 2: BpLayer2, 3: BpLayer3, 4: BpLayer4}


def make_timedeltas_from_zero(look_days: list[float]) -> list[dt.timedelta]:
    return [dt.timedelta(days=lookbehind_day) for lookbehind_day in look_days]


class BpFeatureSpecifier:
    """Feature specification class."""

    def _get_outcome_specs(self) -> list[ValueSpecification]:
        log.info("-------- Generating outcome specs --------")

        return [
            OutcomeSpec(
                value_frame=ValueFrame(
                    init_df=get_first_bipolar_diagnosis(),
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="timestamp",
                ),
                lookahead_distances=make_timedeltas_from_zero(
                    look_days=[10_000]
                ),
                fallback=0,
                aggregators=[MaxAggregator()],
            )
        ]

    def get_feature_specs(
        self, max_layer: int, lookbehind_days: list[float]
    ) -> list[ValueSpecification]:
        if max_layer not in BpFeatureLayers:
            raise ValueError(f"Layer {max_layer} not supported.")

        feature_specs: list[Sequence[ValueSpecification]] = [            
            self._get_outcome_specs(),
        ]


        for layer in range(1, max_layer + 1):
            feature_specs.append(
                BpFeatureLayers[layer]().get_features(lookbehind_days=lookbehind_days)
            )

        # Flatten the Sequence of lists
        features = [feature for sublist in feature_specs for feature in sublist]
        return features
