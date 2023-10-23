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

from psycop.common.feature_generation.loaders.raw.load_demographic import sex_female
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    CVDCohortDefiner,
)
from psycop.projects.cvd.feature_generation.feature_layeres.base import (
    AnySpecType,
)
from psycop.projects.cvd.feature_generation.feature_layeres.layer_1 import CVDLayer1
from psycop.projects.cvd.feature_generation.feature_layeres.layer_2 import CVDLayer2
from psycop.projects.cvd.feature_generation.feature_layeres.layer_3 import CVDLayer3

if TYPE_CHECKING:
    from collections.abc import Sequence

    pass

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
                    df=CVDCohortDefiner.get_outcome_timestamps().to_pandas(),
                    name="score2_cvd",
                ),
            ],
            lookahead_days=[365 * 5],
            aggregation_fns=[maximum],
            fallback=[0],
            incident=[True],
        ).create_combinations()

    def get_feature_specs(self, layer: int) -> list[AnySpecType]:
        """Get a spec set."""
        feature_sequences: list[Sequence[AnySpecType]] = [
            self._get_static_predictor_specs(),
            self._get_outcome_specs(),
        ]

        if layer >= 1:
            feature_sequences.append(CVDLayer1().get_features(lookbehind_days=730))
        if layer >= 2:
            feature_sequences.append(CVDLayer2().get_features(lookbehind_days=730))
        if layer >= 3:
            feature_sequences.append(CVDLayer3().get_features(lookbehind_days=730))
        if layer > 3:
            raise ValueError(f"Layer {layer} not supported.")

        # Flatten the Sequence of lists
        features = [feature for sublist in feature_sequences for feature in sublist]

        return features


if __name__ == "__main__":
    outcome_specs = CVDFeatureSpecifier()._get_outcome_specs()
    outcome_df = outcome_specs[0].timeseries_df

    pass
