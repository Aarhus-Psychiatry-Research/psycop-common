"""Feature specification module."""
import logging
from typing import TYPE_CHECKING

from timeseriesflattener.aggregation_fns import maximum
from timeseriesflattener.feature_specs.group_specs import NamedDataframe, OutcomeGroupSpec
from timeseriesflattener.feature_specs.single_specs import AnySpec, OutcomeSpec, StaticSpec

from psycop.common.feature_generation.loaders.raw.load_visits import (
    get_time_of_first_visit_to_psychiatry,
)
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_prediction_time_loader import (
    SczBpCohort,
)
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_layer_1 import SczBpLayer1
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_layer_2 import SczBpLayer2
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_layer_3 import SczBpLayer3
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_layer_4 import SczBpLayer4
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_layer_5 import SczBpLayer5
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_layer_6 import SczBpLayer6
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_layer_7 import SczBpLayer7
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
    5: SczBpLayer5,
    6: SczBpLayer6,
    7: SczBpLayer7,
}


class SczBpFeatureSpecifier:
    """Feature specification class."""

    def _get_outcome_specs(self) -> list[OutcomeSpec]:
        log.info("-------- Generating outcome specs --------")

        return OutcomeGroupSpec(
            named_dataframes=[
                NamedDataframe(
                    df=SczBpCohort.get_outcome_timestamps().frame.to_pandas(),
                    name="first_scz_or_bp",
                )
            ],
            lookahead_days=[year * 365 for year in (1, 2, 3, 4, 5)],
            aggregation_fns=[maximum],
            fallback=[0],
            incident=[True],
        ).create_combinations()

    def _get_metadata_specs(self) -> list[AnySpec]:
        log.info("-------- Generating metadata specs --------")

        return [
            StaticSpec(
                feature_base_name="scz_or_bp_indicator",
                timeseries_df=get_diagnosis_type_of_first_scz_bp_diagnosis_after_washin().to_pandas(),
                prefix="meta",
            ),
            StaticSpec(
                feature_base_name="time_of_diagnosis",
                timeseries_df=get_time_of_first_scz_or_bp_diagnosis_after_washin().to_pandas(),
                prefix="meta",
            ),
            StaticSpec(
                feature_base_name="first_visit",
                timeseries_df=get_time_of_first_visit_to_psychiatry().to_pandas(),
                prefix="meta",
            ),
            OutcomeSpec(
                feature_base_name="scz_within_3_years",
                timeseries_df=get_first_scz_diagnosis(),
                lookahead_days=1095,
                aggregation_fn=maximum,
                fallback=0,
                incident=True,
                prefix="meta",
            ),
            OutcomeSpec(
                feature_base_name="bp_within_3_years",
                timeseries_df=get_first_bp_diagnosis(),
                lookahead_days=1095,
                aggregation_fn=maximum,
                fallback=0,
                incident=True,
                prefix="meta",
            ),
        ]

    def get_feature_specs(self, max_layer: int, lookbehind_days: list[float]) -> list[AnySpec]:
        if max_layer not in SczBpFeatureLayers:
            raise ValueError(f"Layer {max_layer} not supported.")

        feature_specs: list[Sequence[AnySpec]] = [
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
