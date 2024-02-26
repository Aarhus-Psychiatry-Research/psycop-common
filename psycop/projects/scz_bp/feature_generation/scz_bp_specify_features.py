"""Feature specification module."""
import datetime as dt
import logging
from typing import TYPE_CHECKING, Union

import numpy as np
from timeseriesflattener import (
    BooleanOutcomeSpec,
    OutcomeSpec,
    PredictorSpec,
    StaticFrame,
    StaticSpec,
    TimeDeltaSpec,
    TimestampValueFrame,
    ValueFrame,
)
from timeseriesflattener.aggregators import MaxAggregator
from timeseriesflattener.v1.aggregation_fns import maximum
from timeseriesflattener.v1.feature_specs.group_specs import (
    NamedDataframe,
    OutcomeGroupSpec,
)
from timeseriesflattener.v1.feature_specs.single_specs import AnySpec

from psycop.common.feature_generation.loaders.raw.load_visits import (
    get_time_of_first_visit_to_psychiatry,
)
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_prediction_time_loader import (
    SczBpCohort,
)
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_layer_1 import (
    SczBpLayer1,
)
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_layer_2 import (
    SczBpLayer2,
)
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_layer_3 import (
    SczBpLayer3,
)
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_layer_4 import (
    SczBpLayer4,
)

from psycop.projects.scz_bp.feature_generation.feature_layers.value_specification import (
    ValueSpecification,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.bp_diagnoses import (
    get_first_bp_diagnosis,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.first_scz_or_bp_diagnosis import (
    get_diagnosis_type_of_first_scz_bp_diagnosis_after_washin,
    get_time_of_first_scz_or_bp_diagnosis_after_washin,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.scz_diagnoses import (
    get_first_scz_diagnosis,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger(__name__)


SczBpFeatureLayers = {
    1: SczBpLayer1,
    2: SczBpLayer2,
    3: SczBpLayer3,
    4: SczBpLayer4,
}



def make_timedeltas_from_zero(look_days: list[float]) -> list[dt.timedelta]:
    return [dt.timedelta(days=lookbehind_day) for lookbehind_day in look_days]


class SczBpFeatureSpecifier:
    """Feature specification class."""

    def _get_outcome_specs(self) -> list[ValueSpecification]:
        log.info("-------- Generating outcome specs --------")

        return [
            OutcomeSpec(
                value_frame=ValueFrame(
                    init_df=SczBpCohort.get_outcome_timestamps().frame.rename(
                        {"value": "first_scz_or_bp"}
                    ),
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="timestamp",
                ),
                lookahead_distances=make_timedeltas_from_zero(
                    look_days=[year * 365 for year in (1, 2, 3, 4, 5)]
                ),
                fallback=0,
                aggregators=[MaxAggregator()],
                column_prefix="outc_first_scz_or_bp",
            )
        ]

    def _get_metadata_specs(self) -> list[ValueSpecification]:
        log.info("-------- Generating metadata specs --------")

        return [
            StaticSpec(
                value_frame=StaticFrame(
                    init_df=get_diagnosis_type_of_first_scz_bp_diagnosis_after_washin().rename(
                        {"value": "scz_or_bp_indicator"}
                    ),
                    entity_id_col_name="dw_ek_borger",
                ),
                column_prefix="meta",
                fallback=np.nan,
            ),
            StaticSpec(
                value_frame=StaticFrame(
                    init_df=get_time_of_first_scz_or_bp_diagnosis_after_washin().rename(
                        {"value": "time_of_diagnosis"}
                    ),
                    entity_id_col_name="dw_ek_borger",
                ),
                column_prefix="meta",
                fallback=np.nan,
            ),
            StaticSpec(
                value_frame=StaticFrame(
                    init_df=get_time_of_first_visit_to_psychiatry().rename(
                        {"value": "first_visit"}
                    ),
                    entity_id_col_name="dw_ek_borger",
                ),
                column_prefix="meta",
                fallback=np.nan,
            ),
            OutcomeSpec(
                value_frame=ValueFrame(
                    init_df=get_first_scz_diagnosis().rename({"value": "scz_within_3_years"}),
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="timestamp",
                ),
                lookahead_distances=make_timedeltas_from_zero(look_days=[1095]),
                aggregators=[MaxAggregator()],
                fallback=0,
                column_prefix="meta",
            ),
            OutcomeSpec(
                value_frame=ValueFrame(
                    init_df=get_first_bp_diagnosis().rename({"value": "bp_within_3_years"}),
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="timestamp",
                ),
                lookahead_distances=make_timedeltas_from_zero(look_days=[1095]),
                aggregators=[MaxAggregator()],
                fallback=0,
                column_prefix="meta",
            ),
        ]

    def get_feature_specs(self, max_layer: int, lookbehind_days: list[float]) -> list[ValueSpecification]:
        if max_layer not in SczBpFeatureLayers:
            raise ValueError(f"Layer {max_layer} not supported.")

        feature_specs: list[Sequence[ValueSpecification]] = [
            self._get_metadata_specs(),
            self._get_outcome_specs(),
        ]

        for layer in range(1, max_layer + 1):
            feature_specs.append(
                SczBpFeatureLayers[layer]().get_features(lookbehind_days=lookbehind_days)
            )

        # Flatten the Sequence of lists
        features = [feature for sublist in feature_specs for feature in sublist]
        return features
