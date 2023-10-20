"""Feature specification module."""
import logging
from typing import TYPE_CHECKING

from timeseriesflattener.aggregation_fns import (
    maximum,
)
from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
    OutcomeGroupSpec,
)
from timeseriesflattener.feature_specs.single_specs import (
    AnySpec,
    BaseModel,
    OutcomeSpec,
    PredictorSpec,
    StaticSpec,
)

from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from psycop.common.feature_generation.loaders.raw.load_demographic import sex_female
from psycop.common.feature_generation.loaders.raw.load_diagnoses import (
    SCORE2_CVD,
)
from psycop.projects.cvd.feature_generation.feature_layeres.layer_1 import CVDLayer1
from psycop.projects.cvd.feature_generation.feature_layeres.layer_2 import CVDLayer2
from psycop.projects.cvd.feature_generation.feature_layeres.layer_3 import CVDLayer3

if TYPE_CHECKING:
    from psycop.projects.cvd.feature_generation.feature_layeres.base import (
        FeatureLayer,
    )

log = logging.getLogger(__name__)


class SpecSet(BaseModel):
    """A set of unresolved specs, ready for resolving."""

    temporal_predictors: list[PredictorSpec]
    static_predictors: list[StaticSpec]
    outcomes: list[OutcomeSpec]
    metadata: list[AnySpec]


class CVDFeatureSpecifier:
    """Feature specification class."""

    def _get_static_predictor_specs(self) -> list[StaticSpec]:
        """Get static predictor specs."""
        return [
            StaticSpec(
                feature_base_name="sex_female",
                timeseries_df=sex_female(),
            ),
        ]

    def _get_outcome_specs(self) -> list[OutcomeSpec]:
        """Get outcome specs."""
        log.info("-------- Generating outcome specs --------")

        return OutcomeGroupSpec(
            named_dataframes=[
                NamedDataframe(
                    df=SCORE2_CVD(),
                    name="score2_cvd",
                ),
            ],
            lookahead_days=[365 * 5],
            aggregation_fns=[maximum],
            fallback=[0],
            incident=[True],
        ).create_combinations()

    def get_feature_specs(self, layer: int) -> list[AnySpec]:
        """Get a spec set."""

        layers: list[FeatureLayer] = [CVDLayer1()]

        if layer >= 2:
            layers.append(CVDLayer2())
        if layer >= 3:
            layers.append(CVDLayer3())
        if layer > 3:
            raise ValueError(f"Layer {layer} not supported.")

        temporal_predictor_sequences = [
            layer.get_features(lookbehind_days=730) for layer in layers
        ]
        temporal_predictors = [
            item for sublist in temporal_predictor_sequences for item in sublist
        ]

        return (
            temporal_predictors
            + self._get_static_predictor_specs()
            + self._get_outcome_specs()
        )
