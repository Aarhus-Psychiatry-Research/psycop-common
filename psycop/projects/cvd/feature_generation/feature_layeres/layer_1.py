from collections.abc import Sequence

import numpy as np
from timeseriesflattener.aggregation_fns import AggregationFunType
from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
    PredictorGroupSpec,
    PredictorSpec,
)

from psycop.common.feature_generation.loaders.raw.load_lab_results import ldl
from psycop.common.feature_generation.loaders.raw.load_structured_sfi import (
    smoking_categorical,
    smoking_continuous,
    systolic_blood_pressure,
)
from psycop.projects.cvd.feature_generation.feature_layeres.base import FeatureLayer


class CVDLayer1(FeatureLayer):
    def get_features(
        self, lookbehind_days: Sequence[int], aggregation_fns: Sequence[AggregationFunType]
    ) -> Sequence[PredictorSpec]:
        layer = 1
        ldl_spec = PredictorGroupSpec(
            named_dataframes=(NamedDataframe(df=ldl(), name=f"ldl_layer_{layer}"),),
            aggregation_fns=aggregation_fns,
            fallback=[np.nan],
            lookbehind_days=lookbehind_days,
        ).create_combinations()

        # systolic_blood_pressure_spec = PredictorGroupSpec(
        #     named_dataframes=(
        #         NamedDataframe(
        #             df=systolic_blood_pressure(), name=f"systolic_blood_pressure_layer_{layer}"
        #         ),
        #     ),
        #     aggregation_fns=aggregation_fns,
        #     fallback=[np.nan],
        #     lookbehind_days=lookbehind_days,
        # ).create_combinations()

        smoking_continuous_spec = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=smoking_continuous(), name=f"smoking_continuous_layer_{layer}"),
            ),
            aggregation_fns=aggregation_fns,
            fallback=[np.nan],
            lookbehind_days=lookbehind_days,
        ).create_combinations()

        smoking_categorical_spec = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=smoking_categorical(), name=f"smoking_categorical_layer_{layer}"),
            ),
            aggregation_fns=aggregation_fns,
            fallback=[np.nan],
            lookbehind_days=lookbehind_days,
        ).create_combinations()

        return (
            ldl_spec
            # + systolic_blood_pressure_spec
            + smoking_continuous_spec
            + smoking_categorical_spec
        )
