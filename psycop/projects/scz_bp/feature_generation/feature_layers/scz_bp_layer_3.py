from collections.abc import Sequence

import numpy as np
from timeseriesflattener.aggregation_fns import count, latest
from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
    PredictorGroupSpec,
)
from timeseriesflattener.feature_specs.single_specs import (
    AnySpec,
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
from psycop.common.feature_generation.loaders.raw.load_structured_sfi import (
    broeset_violence_checklist,
    hamilton_d17,
)
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_feature_layer import (
    SczBpFeatureLayer,
)


class SczBpLayer3(SczBpFeatureLayer):
    def get_features(self, lookbehind_days: list[float]) -> Sequence[AnySpec]:
        layer = 3

        psychiatric_diagnoses = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=f0_disorders(), name="f0_disorders"),
                NamedDataframe(df=f1_disorders(), name="f1_disorders"),
                NamedDataframe(df=f2_disorders(), name="f2_disorders"),
                NamedDataframe(df=f3_disorders(), name="f3_disorders"),
                NamedDataframe(df=f4_disorders(), name="f4_disorders"),
                NamedDataframe(df=f5_disorders(), name="f5_disorders"),
                NamedDataframe(df=f6_disorders(), name="f6_disorders"),
                NamedDataframe(df=f7_disorders(), name="f7_disorders"),
                NamedDataframe(df=f8_disorders(), name="f8_disorders"),
                NamedDataframe(df=f9_disorders(), name="f9_disorders"),
            ),
            lookbehind_days=lookbehind_days,
            aggregation_fns=[count],
            fallback=[0],
        ).create_combinations()

        hamilton_spec = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(
                    df=hamilton_d17(),
                    name=f"hamilton_d17_layer_{layer}",
                ),
            ),
            lookbehind_days=lookbehind_days,
            aggregation_fns=[latest],
            fallback=[np.nan],
        ).create_combinations()

        broeset_violence_spec = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(
                    df=broeset_violence_checklist(),
                    name=f"broeset_violence_checklist_layer_{layer}",
                ),
            ),
            lookbehind_days=lookbehind_days,
            aggregation_fns=[latest],
            fallback=[np.nan],
        ).create_combinations()

        return psychiatric_diagnoses + hamilton_spec + broeset_violence_spec
