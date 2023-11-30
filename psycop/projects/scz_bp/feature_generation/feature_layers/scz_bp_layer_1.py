from collections.abc import Sequence

from timeseriesflattener.aggregation_fns import count
from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
    PredictorGroupSpec,
)
from timeseriesflattener.feature_specs.single_specs import (
    AnySpec,
    StaticSpec,
)

from psycop.common.feature_generation.loaders.raw.load_demographic import sex_female
from psycop.common.feature_generation.loaders.raw.load_medications import (
    antidepressives,
    antipsychotics,
)
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_feature_layer import (
    SczBpFeatureLayer,
)


class SczBpLayer1(SczBpFeatureLayer):
    def get_features(self, lookbehind_days: list[float]) -> Sequence[AnySpec]:
        layer = 1

        sex_spec = [
            StaticSpec(
                feature_base_name=f"sex_female_layer_{layer}",
                timeseries_df=sex_female(),
            ),
        ]

        antipsychotics_spec = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(
                    df=antipsychotics(),
                    name=f"antipsychotics_layer_{layer}",
                ),
            ),
            lookbehind_days=lookbehind_days,
            aggregation_fns=[count],
            fallback=[0],
        ).create_combinations()

        antidepressives_spec = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(
                    df=antidepressives(),
                    name=f"antidepressives_layer_{layer}",
                ),
            ),
            lookbehind_days=lookbehind_days,
            aggregation_fns=[count],
            fallback=[0],
        ).create_combinations()

        return sex_spec + antipsychotics_spec + antidepressives_spec
