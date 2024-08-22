from collections.abc import Sequence

import numpy as np
from timeseriesflattener import PredictorGroupSpec
from timeseriesflattener.v1.aggregation_fns import boolean, latest
from timeseriesflattener.v1.feature_specs.group_specs import NamedDataframe

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
from psycop.projects.scz_bp.feature_generation.feature_layers.value_specification import (
    ValueSpecification,
)


class SczBpLayer3(SczBpFeatureLayer):
    def get_features(self, lookbehind_days: list[float]) -> Sequence[ValueSpecification]:
        layer = 3

        psychiatric_diagnoses = list(
            PredictorGroupSpec(
                named_dataframes=(
                    NamedDataframe(df=f0_disorders(), name=f"f0_disorders_layer_{layer}"),
                    NamedDataframe(df=f1_disorders(), name=f"f1_disorders_layer_{layer}"),
                    NamedDataframe(df=f2_disorders(), name=f"f2_disorders_layer_{layer}"),
                    NamedDataframe(df=f3_disorders(), name=f"f3_disorders_layer_{layer}"),
                    NamedDataframe(df=f4_disorders(), name=f"f4_disorders_layer_{layer}"),
                    NamedDataframe(df=f5_disorders(), name=f"f5_disorders_layer_{layer}"),
                    NamedDataframe(df=f6_disorders(), name=f"f6_disorders_layer_{layer}"),
                    NamedDataframe(df=f7_disorders(), name=f"f7_disorders_layer_{layer}"),
                    NamedDataframe(df=f8_disorders(), name=f"f8_disorders_layer_{layer}"),
                    NamedDataframe(df=f9_disorders(), name=f"f9_disorders_layer_{layer}"),
                ),
                lookbehind_days=lookbehind_days,
                aggregation_fns=[boolean],
                fallback=[0],
                entity_id_col_name_out="dw_ek_borger",
            ).create_combinations()
        )

        hamilton_spec = list(
            PredictorGroupSpec(
                named_dataframes=(
                    NamedDataframe(df=hamilton_d17(), name=f"hamilton_d17_layer_{layer}"),
                ),
                lookbehind_days=lookbehind_days,
                aggregation_fns=[latest],
                fallback=[np.nan],
                entity_id_col_name_out="dw_ek_borger",
            ).create_combinations()
        )

        broeset_violence_spec = list(
            PredictorGroupSpec(
                named_dataframes=(
                    NamedDataframe(
                        df=broeset_violence_checklist(),
                        name=f"broeset_violence_checklist_layer_{layer}",
                    ),
                ),
                lookbehind_days=lookbehind_days,
                aggregation_fns=[latest],
                fallback=[np.nan],
                entity_id_col_name_out="dw_ek_borger",
            ).create_combinations()
        )

        return psychiatric_diagnoses + hamilton_spec + broeset_violence_spec
