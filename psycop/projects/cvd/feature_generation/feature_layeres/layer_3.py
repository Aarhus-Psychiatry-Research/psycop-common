from collections.abc import Sequence

from timeseriesflattener.aggregation_fns import boolean, count
from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
    PredictorGroupSpec,
    PredictorSpec,
)

from psycop.common.feature_generation.loaders.raw.load_diagnoses import (
    f0_disorders,
    f1_disorders,
    f2_disorders,
    f3_disorders,
    f4_disorders,
    f5_disorders,
    f6_disorders,
    f7_disorders,
    f8_disorders,
    f9_disorders,
)
from psycop.common.feature_generation.loaders.raw.load_lab_results import hdl
from psycop.common.feature_generation.loaders.raw.load_medications import (
    top_10_weight_gaining_antipsychotics,
)
from psycop.projects.cvd.feature_generation.feature_layeres.base import (
    FeatureLayer,
)


class CVDLayer3(FeatureLayer):
    def get_features(self, lookbehind_days: int) -> Sequence[PredictorSpec]:
        layer = 3
        psychiatric_disorders = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(
                    df=f0_disorders(),
                    name=f"f0_disorders_layer_layer_{layer}",
                ),
                NamedDataframe(
                    df=f1_disorders(),
                    name=f"f1_disorders_layer_{layer}",
                ),
                NamedDataframe(
                    df=f2_disorders(),
                    name=f"f2_disorders_layer_{layer}",
                ),
                NamedDataframe(
                    df=f3_disorders(),
                    name=f"f3_disorders_layer_{layer}",
                ),
                NamedDataframe(
                    df=f4_disorders(),
                    name=f"f4_disorders_layer_{layer}",
                ),
                NamedDataframe(
                    df=f5_disorders(),
                    name=f"f5_disorders_layer_{layer}",
                ),
                NamedDataframe(
                    df=f6_disorders(),
                    name=f"f6_disorders_layer_{layer}",
                ),
                NamedDataframe(
                    df=f7_disorders(),
                    name=f"f7_disorders_layer_{layer}",
                ),
                NamedDataframe(
                    df=f8_disorders(),
                    name=f"f8_disorders_layer_{layer}",
                ),
                NamedDataframe(
                    df=f9_disorders(),
                    name=f"f9_disorders_layer_{layer}",
                ),
                NamedDataframe(
                    df=top_10_weight_gaining_antipsychotics(),
                    name=f"top_10_weight_gaining_antipsychotics_layer_{layer}",
                ),
            ),
            aggregation_fns=[count],
            lookbehind_days=[lookbehind_days],
            fallback=[0],
        ).create_combinations()

        antipsychotics_spec = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(
                    df=top_10_weight_gaining_antipsychotics(),
                    name=f"antipsychotics_layer_{layer}",
                ),
            ),
            lookbehind_days=[lookbehind_days],
            aggregation_fns=[boolean],
            fallback=[0],
        ).create_combinations()

        hdl_spec = PredictorGroupSpec(
            named_dataframes=[NamedDataframe(df=hdl(), name=f"hdl_layer_{layer}")],
            lookbehind_days=[lookbehind_days],
            aggregation_fns=[boolean],
            fallback=[0],
        ).create_combinations()

        return psychiatric_disorders + antipsychotics_spec + hdl_spec
